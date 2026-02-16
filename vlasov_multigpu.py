#%%
"Multi-GPU Vlasov solver using JAX sharding"

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import matplotlib.pyplot as plt
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from tqdm import tqdm
import numpy as np
from scipy.signal import argrelextrema

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------
devices = jax.devices()
num_devices = len(devices)
mesh = Mesh(np.array(devices), axis_names=('devices',))

# Sharding specs
sharded  = NamedSharding(mesh, P('devices'))       # shard dim-0 across devices
sharded2 = NamedSharding(mesh, P('devices', None)) # shard dim-0, replicate dim-1
replicated = NamedSharding(mesh, P())               # full replicate

print(f"JAX backend : {jax.default_backend()}")
print(f"Devices ({num_devices}): {devices}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def visualize_initial(x, v, cells, E, rho, eta, L,
                      v_target=lambda v: jax.scipy.stats.norm.pdf(v, 0, 1)):
    """Visualize initial data."""
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].hist(np.asarray(x), bins=50, density=True, alpha=0.4, label='Sampled $x$')
    x_grid = jnp.linspace(0, L, 200)
    axs[0].plot(x_grid, spatial_density(x_grid), 'r-', label='Target density')
    axs[0].plot(cells, rho / L, 'g-', label='$\\rho/L$')
    axs[0].set_title('Position $x$')
    axs[0].set_xlabel('$x$')
    axs[0].legend()

    axs[1].hist(np.asarray(v), bins=50, density=True, alpha=0.4, label='Sampled $v$')
    v_grid = jnp.linspace(float(jnp.min(v))-1, float(jnp.max(v))+1, 200)
    axs[1].plot(v_grid, v_target(v_grid), 'r-', label='Target $N(0,1)$')
    axs[1].set_title('Velocity $v$')
    axs[1].set_xlabel('$v$')
    axs[1].legend()

    axs[2].plot(cells, E, label='$E$')
    dE_dx = jnp.gradient(E, eta)
    axs[2].plot(cells, dE_dx, label='$dE/dx$')
    axs[2].plot(cells, rho - 1, label=r'$\rho - \rho_i$')
    axs[2].set_title('Field $E$, $dE/dx$, and $\\rho$')
    axs[2].set_xlabel('x')
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def rejection_sample(key, density_fn, domain, max_value, num_samples=1):
    "Sample in parallel via rejection sampling."
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
# Core PIC kernels  (multi-device via shard_map)
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
        """One forward-Euler PIC step (multi-device)."""
        E_at_p = evaluate_field_at_particles(E, x)
        v_new  = v.at[:, 0].add(dt_val * E_at_p)
        x_new  = jnp.mod(x + dt_val * v[:, 0], box_length)
        E_new  = update_electric_field(E, x, v)
        return x_new, v_new, E_new

    return pic_step, evaluate_charge_density


# ---------------------------------------------------------------------------
# Main simulation  — Landau damping
# ---------------------------------------------------------------------------
#%%
seed = 42

q  = 1
dv = 1

alpha = 0.1
k     = 0.5
L     = 2 * jnp.pi / k
n     = 10**8
M     = 100
dt    = 0.02
print(f"Running n={n:.0e}, M={M}, dt={dt}")

eta   = L / M
cells = (jnp.arange(M) + 0.5) * eta
w     = q * L / n

# --- initial velocity ---
key_v, key_x = jr.split(jr.PRNGKey(seed), 2)
v = jr.multivariate_normal(key_v, jnp.zeros(dv), jnp.eye(dv), shape=(n,)).reshape((n, dv))
v = v - jnp.mean(v, axis=0)

# --- initial positions (rejection sampling, then shard) ---
def spatial_density(x):
    return (1 + alpha * jnp.cos(k * x)) / (2 * jnp.pi / k)

max_value = jnp.max(spatial_density(cells))
domain = (0, L)
x = rejection_sample(key_x, spatial_density, domain, max_value=max_value, num_samples=n)

# --- build jitted PIC step (closes over M, eta, w, dt, L) ---
pic_step, evaluate_charge_density = make_pic_step(mesh, M, eta, w, dt, L)

# --- initial electric field ---
rho_init = evaluate_charge_density(x)
E = jnp.cumsum(rho_init - 1) * eta
E = E - jnp.mean(E)

visualize_initial(x, v[:, 0], cells, E, rho_init, eta, L)

# --- distribute arrays across devices ---
# Pad particle count up to multiple of num_devices for clean sharding
pad = (-n) % num_devices
if pad > 0:
    x = jnp.concatenate([x, jnp.zeros(pad)])
    v = jnp.concatenate([v, jnp.zeros((pad, dv))])
n_padded = n + pad

x = jax.device_put(x, sharded)
v = jax.device_put(v, sharded2)
E = jax.device_put(E, replicated)
cells = jax.device_put(cells, replicated)

print(f"Particles per device: {n_padded // num_devices:,}")

# --- time integration ---
final_time = 30.0
num_steps  = int(final_time / dt)
t = 0.0
E_L2 = [float(jnp.sqrt(jnp.sum(E**2) * eta))]

for step_num in tqdm(range(num_steps)):
    x, v, E = pic_step(x, v, E)
    E = E - jnp.mean(E)  # enforce zero-mean
    t += dt
    E_L2.append(float(jnp.sqrt(jnp.sum(E**2) * eta)))

# ---------------------------------------------------------------------------
# Diagnostics / plotting
# ---------------------------------------------------------------------------
#%%
plt.figure(figsize=(6, 4))
t_grid = np.linspace(0, final_time, num_steps + 1)
plt.plot(t_grid, E_L2, marker='o', markersize=1, label='Simulation')

# Predicted curve
prefactor = -1 / (k**3) * np.sqrt(np.pi / 8) * np.exp(-1 / (2 * k**2) - 1.5)
predicted = np.exp(t_grid * prefactor)
predicted *= E_L2[0] / predicted[0]
gamma = prefactor
plt.plot(t_grid, predicted, 'r--', label=fr'$e^{{\gamma t}}, \gamma = {gamma:.3f}$')

# Fit in log space
E_L2_np = np.asarray(E_L2)
mask = (t_grid > 0.2) & (t_grid < 15)
t_mask = t_grid[mask]
n_mask = E_L2_np[mask]

maxima_indices = argrelextrema(n_mask, np.greater, order=5)[0]
mt = t_mask[maxima_indices]
mv = n_mask[maxima_indices]
plt.scatter(mt, mv, color='g', marker='o', zorder=5)
coeffs = np.polyfit(mt, np.log(mv), 1)
fit = np.exp(coeffs[1] + coeffs[0] * t_mask)
plt.plot(t_mask, fit, 'g--', label=fr'$e^{{\beta t}}, \beta={coeffs[0]:.3f}$')

plt.xlabel('Time')
plt.ylabel(r'$||E||_{L^2}$')
plt.title(f"n={n:.0e}, Δt={dt}, dv={dv}, α={alpha}, C=0, M={M}")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
