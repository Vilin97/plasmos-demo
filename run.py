"""Runnable PIC Vlasov solver with multi-GPU support."""

import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from scipy.signal import argrelextrema
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rejection_sample(key, density_fn, domain, max_value, num_samples=1):
    domain_width = domain[1] - domain[0]
    max_ratio = max_value / (1.0 / domain_width) * 1.2
    key, key_propose, key_accept = jr.split(key, 3)
    num_candidates = int(num_samples * max_ratio * 2)
    candidates = jr.uniform(key_propose, minval=domain[0], maxval=domain[1],
                            shape=(num_candidates,))
    proposal_values = jnp.where(
        (candidates >= domain[0]) & (candidates <= domain[1]),
        1.0 / domain_width, 0.0
    )
    target_values = density_fn(candidates)
    accepted = (jr.uniform(key_accept, (num_candidates,))
                * max_ratio * proposal_values <= target_values)
    return candidates[accepted][:num_samples]


# ---------------------------------------------------------------------------
# PIC kernels (multi-device via shard_map)
# ---------------------------------------------------------------------------
def make_pic_step(mesh, M, eta, w, dt_val, box_length):
    """Build a jitted PIC step that closes over static parameters.
    Returns (pic_step, evaluate_charge_density)."""

    def evaluate_charge_density(x):
        def _local_rho(x_local):
            idx_f = x_local / eta - 0.5
            i0    = jnp.floor(idx_f).astype(jnp.int32) % M
            i1    = (i0 + 1) % M
            f     = idx_f - jnp.floor(idx_f)
            counts = jnp.zeros(M).at[i0].add(1 - f).at[i1].add(f)
            counts = lax.psum(counts, 'devices')
            return w / eta * counts
        return shard_map(
            _local_rho, mesh=mesh,
            in_specs=(P('devices'),), out_specs=P(),
            check_rep=False,
        )(x)

    def evaluate_field_at_particles(E, x):
        def _local_interp(E_rep, x_local):
            idx_f = x_local / eta - 0.5
            i0    = jnp.floor(idx_f).astype(jnp.int32) % M
            f     = idx_f - jnp.floor(idx_f)
            i1    = (i0 + 1) % M
            return (1.0 - f) * E_rep[i0] + f * E_rep[i1]
        return shard_map(
            _local_interp, mesh=mesh,
            in_specs=(P(), P('devices')), out_specs=P('devices'),
            check_rep=False,
        )(E, x)

    def update_electric_field(E, x, v):
        def _local_current(E_rep, x_local, v_local):
            idx_f = x_local / eta - 0.5
            i0    = jnp.floor(idx_f).astype(jnp.int32) % M
            i1    = (i0 + 1) % M
            f     = idx_f - jnp.floor(idx_f)
            J = jnp.zeros(M).at[i0].add((1 - f) * v_local[:, 0]).at[i1].add(f * v_local[:, 0])
            J = lax.psum(J, 'devices')
            dEdt = w / eta * J
            return (E_rep - dt_val * dEdt).astype(E_rep.dtype)
        return shard_map(
            _local_current, mesh=mesh,
            in_specs=(P(), P('devices'), P('devices', None)),
            out_specs=P(), check_rep=False,
        )(E, x, v)

    @jax.jit
    def pic_step(x, v, E):
        E_at_p = evaluate_field_at_particles(E, x)
        v_new  = v.at[:, 0].add(dt_val * E_at_p)
        x_new  = jnp.mod(x + dt_val * v[:, 0], box_length)
        E_new  = update_electric_field(E, x, v)
        return x_new, v_new, E_new

    return pic_step, evaluate_charge_density


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PIC Vlasov solver (multi-GPU)")
    parser.add_argument("--dv", type=int, default=1, help="velocity dimensions")
    parser.add_argument("--n", type=int, default=10**6, help="number of particles")
    parser.add_argument("--M", type=int, default=100, help="number of cells")
    parser.add_argument("--dt", type=float, default=0.01, help="time step")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- device setup ---
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('devices',))
    sharded   = NamedSharding(mesh, P('devices'))
    sharded2  = NamedSharding(mesh, P('devices', None))
    replicated = NamedSharding(mesh, P())

    print(f"JAX backend : {jax.default_backend()}")
    print(f"Devices ({num_devices}): {devices}")

    # --- physical parameters ---
    dv = args.dv
    n, M, dt = args.n, args.M, args.dt
    alpha, k = 0.1, 0.5
    L   = float(2 * np.pi / k)
    eta = L / M
    w   = L / n
    cells = (jnp.arange(M) + 0.5) * eta

    print(f"Running dv={dv}, n={n:.0e}, M={M}, dt={dt}")

    # --- build jitted kernels ---
    pic_step, evaluate_charge_density = make_pic_step(mesh, M, eta, w, dt, L)

    # --- initial velocity ---
    key_v, key_x = jr.split(jr.PRNGKey(args.seed), 2)
    v = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(n,)).reshape((n, dv))
    v = v - jnp.mean(v, axis=0)

    # --- initial positions (rejection sampling) ---
    def spatial_density(x):
        return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

    max_value = jnp.max(spatial_density(cells))
    x = rejection_sample(key_x, spatial_density, (0, L),
                         max_value=max_value, num_samples=n)

    # --- initial electric field ---
    rho_init = evaluate_charge_density(x)
    E = jnp.cumsum(rho_init - 1) * eta
    E = E - jnp.mean(E)

    # --- pad & shard ---
    pad = (-n) % num_devices
    if pad > 0:
        x = jnp.concatenate([x, jnp.zeros(pad)])
        v = jnp.concatenate([v, jnp.zeros((pad, dv))])

    x = jax.device_put(x, sharded)
    v = jax.device_put(v, sharded2)
    E = jax.device_put(E, replicated)

    print(f"Particles per device: {(n + pad) // num_devices:,}")

    # --- warmup (compile) ---
    x, v, E = pic_step(x, v, E)
    E = E - jnp.mean(E)
    jax.block_until_ready((x, v, E))

    # --- time integration ---
    final_time = 30.0
    num_steps = int(final_time / dt)
    E_L2 = [float(jnp.sqrt(jnp.sum(E**2) * eta))]

    t0 = time.perf_counter()
    for _ in tqdm(range(num_steps)):
        x, v, E = pic_step(x, v, E)
        E = E - jnp.mean(E)
        E_L2.append(float(jnp.sqrt(jnp.sum(E**2) * eta)))
    jax.block_until_ready((x, v, E))
    runtime = time.perf_counter() - t0

    print(f"Runtime: {runtime:.2f}s  ({runtime / num_steps * 1000:.2f} ms/step)")

    # --- fit damping rate ---
    t_grid = np.linspace(0, final_time, num_steps + 1)
    E_L2_np = np.asarray(E_L2)
    mask = (t_grid > 0.2) & (t_grid < 15)
    t_mask, n_mask = t_grid[mask], E_L2_np[mask]
    maxima_idx = argrelextrema(n_mask, np.greater, order=5)[0]
    mt, mv = t_mask[maxima_idx], n_mask[maxima_idx]

    fitted_slope = None
    coeffs = None
    if len(mt) >= 2:
        coeffs = np.polyfit(mt, np.log(mv), 1)
        fitted_slope = coeffs[0]
        print(f"Fitted slope: {fitted_slope:.4f}")
    else:
        print("Not enough maxima to fit damping rate (need longer final-time)")

    # --- E_L2 plot ---
    prefactor = -1 / (k**3) * np.sqrt(np.pi / 8) * np.exp(-1 / (2 * k**2) - 1.5)
    predicted = np.exp(t_grid * prefactor)
    predicted *= E_L2[0] / predicted[0]

    fig_energy, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t_grid, E_L2_np, marker='o', markersize=1, label='Simulation')
    ax.plot(t_grid, predicted, 'r--',
            label=fr'$e^{{\gamma t}}, \gamma = {prefactor:.3f}$')
    if coeffs is not None:
        fit_curve = np.exp(coeffs[1] + coeffs[0] * t_mask)
        ax.scatter(mt, mv, color='g', marker='o', zorder=5)
        ax.plot(t_mask, fit_curve, 'g--',
                label=fr'$e^{{\beta t}}, \beta={fitted_slope:.3f}$')
    ax.set(xlabel='Time', ylabel=r'$||E||_{L^2}$',
           title=f"n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, C=0, M={M}")
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()
    fig_energy.tight_layout()
    plt.close(fig_energy)

if __name__ == "__main__":
    main()
