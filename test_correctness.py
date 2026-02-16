"""
Correctness test: compare vlasov.py (single-device) vs vlasov_multigpu.py (shard_map).
Both start from identical initial conditions and run for num_steps.
We compare the E_L2 trajectory and final particle/field arrays.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Original single-device kernels (from vlasov.py)
# ---------------------------------------------------------------------------
@jax.jit
def evaluate_charge_density_orig(x, cells, eta, w):
    M      = cells.size
    idx_f  = x / eta - 0.5
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M
    i1     = (i0 + 1) % M
    f      = idx_f - jnp.floor(idx_f)
    counts = jnp.zeros(M).at[i0].add(1 - f).at[i1].add(f)
    return w / eta * counts

@jax.jit
def evaluate_field_at_particles_orig(E, x, cells, eta):
    M      = cells.size
    idx_f  = x / eta - 0.5
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M
    f      = idx_f - jnp.floor(idx_f)
    i1     = (i0 + 1) % M
    return (1.0 - f) * E[i0] + f * E[i1]

@jax.jit
def update_electric_field_orig(E, x, v, cells, eta, w, dt):
    M      = cells.size
    idx_f  = x / eta - 0.5
    i0     = jnp.floor(idx_f).astype(jnp.int32) % M
    i1     = (i0 + 1) % M
    f      = idx_f - jnp.floor(idx_f)
    J = jnp.zeros(M).at[i0].add((1 - f) * v[:, 0]).at[i1].add(f * v[:, 0])
    dEdt = w / eta * J
    return (E - dt * dEdt).astype(E.dtype)

@jax.jit
def step_orig(x, v, E, cells, eta, dt, box_length, w):
    E_at_p = evaluate_field_at_particles_orig(E, x, cells, eta)
    v_new = v.at[:, 0].add(dt * E_at_p)
    x_new = jnp.mod(x + dt * v[:, 0], box_length)
    E_new = update_electric_field_orig(E, x, v, cells, eta, w, dt)
    return x_new, v_new, E_new


# ---------------------------------------------------------------------------
# Multi-device kernels (from vlasov_multigpu.py)
# ---------------------------------------------------------------------------
def make_pic_step(mesh, M, eta, w, dt_val, box_length):
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
# Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=('devices',))
    sharded  = NamedSharding(mesh, P('devices'))
    sharded2 = NamedSharding(mesh, P('devices', None))
    replicated = NamedSharding(mesh, P())

    print(f"Devices: {devices}")

    # Small-ish problem that runs fast
    n = 100_000
    M = 50
    dv = 1
    dt = 0.02
    k = 0.5
    L = float(2 * np.pi / k)
    eta = L / M
    w = L / n
    cells = (jnp.arange(M) + 0.5) * eta
    num_steps = 200

    # Shared initial conditions
    key_v, key_x = jr.split(jr.PRNGKey(42), 2)
    v0 = jr.normal(key_v, shape=(n, dv))
    v0 = v0 - jnp.mean(v0, axis=0)
    x0 = jr.uniform(key_x, shape=(n,), minval=0.0, maxval=L)

    rho0 = evaluate_charge_density_orig(x0, cells, eta, w)
    E0 = jnp.cumsum(rho0 - 1) * eta
    E0 = E0 - jnp.mean(E0)

    # ---- Run original (single-device) ----
    x_s, v_s, E_s = x0.copy(), v0.copy(), E0.copy()
    E_L2_single = [float(jnp.sqrt(jnp.sum(E_s**2) * eta))]
    for _ in range(num_steps):
        x_s, v_s, E_s = step_orig(x_s, v_s, E_s, cells, eta, dt, L, w)
        E_s = E_s - jnp.mean(E_s)
        E_L2_single.append(float(jnp.sqrt(jnp.sum(E_s**2) * eta)))

    # ---- Run multi-device ----
    pic_step, eval_rho_multi = make_pic_step(mesh, M, eta, w, dt, L)

    # Pad for clean sharding
    pad = (-n) % num_devices
    if pad > 0:
        x_m = jnp.concatenate([x0, jnp.zeros(pad)])
        v_m = jnp.concatenate([v0, jnp.zeros((pad, dv))])
    else:
        x_m, v_m = x0.copy(), v0.copy()

    x_m = jax.device_put(x_m, sharded)
    v_m = jax.device_put(v_m, sharded2)
    E_m = jax.device_put(E0.copy(), replicated)

    E_L2_multi = [float(jnp.sqrt(jnp.sum(E_m**2) * eta))]
    for _ in range(num_steps):
        x_m, v_m, E_m = pic_step(x_m, v_m, E_m)
        E_m = E_m - jnp.mean(E_m)
        E_L2_multi.append(float(jnp.sqrt(jnp.sum(E_m**2) * eta)))

    # ---- Compare ----
    E_L2_single = np.array(E_L2_single)
    E_L2_multi  = np.array(E_L2_multi)

    abs_diff = np.abs(E_L2_single - E_L2_multi)
    rel_diff = abs_diff / (np.abs(E_L2_single) + 1e-30)

    print(f"\nE_L2 trajectory ({num_steps} steps, n={n}, M={M}):")
    print(f"  max |diff|     = {abs_diff.max():.2e}")
    print(f"  max rel diff   = {rel_diff.max():.2e}")
    print(f"  mean |diff|    = {abs_diff.mean():.2e}")

    # Compare final field
    E_s_np = np.asarray(E_s)
    E_m_np = np.asarray(E_m)
    E_field_diff = np.max(np.abs(E_s_np - E_m_np))
    print(f"  max |E_final|  = {E_field_diff:.2e}")

    # Compare final positions (only first n, ignoring padding)
    x_s_np = np.asarray(x_s)
    x_m_np = np.asarray(x_m)[:n]
    x_diff = np.max(np.abs(np.sort(x_s_np) - np.sort(x_m_np)))
    print(f"  max |x_final| (sorted) = {x_diff:.2e}")

    tol = 1e-10
    if abs_diff.max() < tol and E_field_diff < tol:
        print(f"\nPASSED: results match to within {tol}")
    else:
        print(f"\nFAILED: differences exceed {tol}")
        # Print first few diverging steps
        bad = np.where(abs_diff > tol)[0]
        if len(bad) > 0:
            print(f"  First divergence at step {bad[0]}:")
            for i in bad[:5]:
                print(f"    step {i}: single={E_L2_single[i]:.15e}  multi={E_L2_multi[i]:.15e}  diff={abs_diff[i]:.2e}")
