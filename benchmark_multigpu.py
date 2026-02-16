"""
Benchmark: 1-GPU vs multi-GPU Vlasov PIC solver.
Measures wall-clock time and per-device peak memory.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import numpy as np
import time
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# PIC kernels (parameterised by mesh + static params closed over)
# ---------------------------------------------------------------------------
def make_kernels(mesh, M, eta, w, dt_val, box_length):
    """Return pic_step and sharding specs.
    M, eta, w, dt_val, box_length are closed over (not traced by jit)."""

    def evaluate_field_at_particles(E, x):
        def _local(E_r, x_l):
            idx_f = x_l / eta - 0.5
            i0 = jnp.floor(idx_f).astype(jnp.int32) % M
            f  = idx_f - jnp.floor(idx_f)
            i1 = (i0 + 1) % M
            return (1.0 - f) * E_r[i0] + f * E_r[i1]
        return shard_map(_local, mesh=mesh,
                         in_specs=(P(), P('devices')), out_specs=P('devices'),
                         check_rep=False)(E, x)

    def update_electric_field(E, x, v):
        def _local(E_r, x_l, v_l):
            idx_f = x_l / eta - 0.5
            i0 = jnp.floor(idx_f).astype(jnp.int32) % M
            i1 = (i0 + 1) % M
            f  = idx_f - jnp.floor(idx_f)
            J = jnp.zeros(M).at[i0].add((1 - f) * v_l[:, 0]).at[i1].add(f * v_l[:, 0])
            J = lax.psum(J, 'devices')
            return (E_r - dt_val * w / eta * J).astype(E_r.dtype)
        return shard_map(_local, mesh=mesh,
                         in_specs=(P(), P('devices'), P('devices', None)),
                         out_specs=P(), check_rep=False)(E, x, v)

    @jax.jit
    def pic_step(x, v, E):
        E_at_p = evaluate_field_at_particles(E, x)
        v_new  = v.at[:, 0].add(dt_val * E_at_p)
        x_new  = jnp.mod(x + dt_val * v[:, 0], box_length)
        E_new  = update_electric_field(E, x, v)
        return x_new, v_new, E_new

    sharded  = NamedSharding(mesh, P('devices'))
    sharded2 = NamedSharding(mesh, P('devices', None))
    replicated = NamedSharding(mesh, P())

    return pic_step, sharded, sharded2, replicated


def get_peak_memory_mb():
    """Return list of peak memory usage in MB per device."""
    result = []
    for d in jax.local_devices():
        stats = d.memory_stats()
        if stats is not None:
            result.append(stats['peak_bytes_in_use'] / 1e6)
        else:
            result.append(float('nan'))
    return result


# ---------------------------------------------------------------------------
# Benchmark one configuration
# ---------------------------------------------------------------------------
def run_benchmark(n_particles, M, dt, dv, num_devices_to_use, num_steps=200):
    """Run PIC for `num_steps` and return (wall_time_s, peak_mem_mb_per_device)."""

    all_devices = jax.devices()
    devices_subset = all_devices[:num_devices_to_use]
    mesh = Mesh(np.array(devices_subset), axis_names=('devices',))

    alpha = 0.1
    k = 0.5
    L = float(2 * np.pi / k)
    eta = L / M
    w = L / n_particles

    pic_step, sharded, sharded2, replicated = make_kernels(mesh, M, eta, w, dt, L)

    # --- init ---
    key_v, key_x = jr.split(jr.PRNGKey(42), 2)
    v = jr.normal(key_v, shape=(n_particles, dv))
    v = v - jnp.mean(v, axis=0)
    x = jr.uniform(key_x, shape=(n_particles,), minval=0.0, maxval=L)

    rho = jnp.zeros(M)
    idx_f = x / eta - 0.5
    i0 = jnp.floor(idx_f).astype(jnp.int32) % M
    i1 = (i0 + 1) % M
    f = idx_f - jnp.floor(idx_f)
    rho = rho.at[i0].add(1 - f).at[i1].add(f)
    rho = w / eta * rho
    E = jnp.cumsum(rho - 1) * eta
    E = E - jnp.mean(E)

    # pad to multiple of device count
    pad = (-n_particles) % num_devices_to_use
    if pad > 0:
        x = jnp.concatenate([x, jnp.zeros(pad)])
        v = jnp.concatenate([v, jnp.zeros((pad, dv))])

    # shard
    x = jax.device_put(x, sharded)
    v = jax.device_put(v, sharded2)
    E = jax.device_put(E, replicated)

    # warmup (compile)
    x, v, E = pic_step(x, v, E)
    E = E - jnp.mean(E)
    jax.block_until_ready((x, v, E))

    mem_before = get_peak_memory_mb()

    t0 = time.perf_counter()
    for _ in range(num_steps):
        x, v, E = pic_step(x, v, E)
        E = E - jnp.mean(E)
    jax.block_until_ready((x, v, E))
    wall = time.perf_counter() - t0

    mem_after = get_peak_memory_mb()

    return wall, mem_after


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    total_devices = len(jax.devices())
    print(f"Total devices: {total_devices}  ({jax.default_backend()})")
    for d in jax.devices():
        stats = d.memory_stats()
        if stats:
            print(f"  {d}: {stats['bytes_limit']/1e9:.1f} GB total")
    print()

    configs = [
        # (n_particles, M,   dt,  dv, num_steps)
        (1_000_000,     100, 0.02, 1,  200),
        (10_000_000,    100, 0.02, 1,  200),
        (100_000_000,   100, 0.02, 1,  100),
    ]

    device_counts = list(range(1, total_devices + 1))

    print(f"{'n':>12s}  {'M':>4s}  {'GPUs':>4s}  {'steps':>5s}  "
          f"{'wall (s)':>9s}  {'step (ms)':>9s}  "
          + "  ".join(f"peak_d{i} (MB)" for i in range(total_devices)))
    print("-" * 100)

    for n_p, M, dt, dv, num_steps in configs:
        for nd in device_counts:
            try:
                jax.clear_caches()
                wall, peaks = run_benchmark(n_p, M, dt, dv, nd, num_steps)
                step_ms = wall / num_steps * 1000
                peak_str = "  ".join(f"{p:>13.1f}" for p in peaks)
                print(f"{n_p:>12,d}  {M:>4d}  {nd:>4d}  {num_steps:>5d}  "
                      f"{wall:>9.2f}  {step_ms:>9.2f}  {peak_str}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"{n_p:>12,d}  {M:>4d}  {nd:>4d}  {num_steps:>5d}  "
                      f"  FAILED: {e}")
        print()
