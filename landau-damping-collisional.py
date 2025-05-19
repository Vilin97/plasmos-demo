#%%
# Landau damping + Landau-collisions (1d-x / 2d-v)  ―  JAX, concise demo
import os, jax, jax.numpy as jnp, jax.random as jr
from tqdm import trange
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# ── parameters ──────────────────────────────────────────────────────────
N, M          = 20_000, 256                      # particles, mesh cells
L             = 4 * jnp.pi                      # domain (k = ½ ⇒ L = 4π)
k, α          = 0.5, 1e-1
dt, T         = 0.1, 10.0                       # time step / final time
steps         = int(T/dt)
dx, η         = L/M, L/M                         # η = dx
w_p           = L/N                              # particle charge ⇒ ∫ρ = L
h_v           = 1.0                              # bandwidth in v
C_col         = 0.1                             # collision strength, try 0, 0.1, 0.5
eps           = 1e-8

# ── initial particles ───────────────────────────────────────────────────
key = jr.PRNGKey(0)
def sample_ic(key):
    key, kx1, kx2, kv1, kv2 = jr.split(key, 5)
    x_cand = jr.uniform(kx1, (int(1.2*N),), minval=0., maxval=L)
    mask   = jr.uniform(kx2, x_cand.shape) < (1+α*jnp.cos(k*x_cand))/(1+α)
    x      = x_cand[mask][:N]
    v1     = jr.normal(kv1, (N,))
    v2     = jr.normal(kv2, (N,))
    return x, v1, v2
x, v1, v2 = sample_ic(key)

# ── helpers ─────────────────────────────────────────────────────────────
hat   = lambda z,h: jnp.maximum(0.0, 1.0 - jnp.abs(z)/h)
dhatz = lambda z,h: jnp.where(jnp.abs(z) < h, -jnp.sign(z)/h, 0.0)

def deposit_rho(x):
    idx = x/dx
    i0  = jnp.floor(idx).astype(jnp.int32) % M
    f   = idx - jnp.floor(idx)
    i1  = (i0+1) % M
    w0, w1 = 1-f, f
    ρ = (jnp.zeros(M)
         .at[i0].add(w_p*w0)
         .at[i1].add(w_p*w1)) / dx
    return ρ

def field_from_rho(ρ):
    δρ = ρ - ρ.mean()
    E  = jnp.cumsum(δρ)*dx
    return E - E.mean()

def interp_E(x,E):
    idx = x/dx
    i0  = jnp.floor(idx).astype(jnp.int32) % M
    f   = idx - jnp.floor(idx); i1 = (i0+1) % M
    return (1-f)*E[i0] + f*E[i1]

# ── KDE score (∇_v log ƒ̂)  and collision operator ──────────────────────
def score_and_collision(x, v1, v2):
    Δx  = (x[:,None] - x[None,:])
    ψx  = hat(Δx, η)
    Δv1 = v1[:,None] - v1[None,:]
    Δv2 = v2[:,None] - v2[None,:]
    ψv1 = hat(Δv1, h_v)
    ψv2 = hat(Δv2, h_v)
    Kv  = ψv1*ψv2
    K   = ψx*Kv                               # (N,N)

    fhat = K.mean(axis=1) + eps               # KDE density

    dψv1 = dhatz(Δv1, h_v)
    dψv2 = dhatz(Δv2, h_v)

    s1 = (ψx*ψv2*dψv1).mean(axis=1) / fhat    # ∇_v log f
    s2 = (ψx*ψv1*dψv2).mean(axis=1) / fhat
    s  = jnp.stack([s1,s2],1)                 # (N,2)

    # --- Landau collision term ------------------------------------------
    dv   = jnp.stack([Δv1, Δv2],-1)           # (N,N,2)
    ds   = s[:,None,:] - s[None,:,:]          # (N,N,2)
    n2   = jnp.sum(dv**2, -1, keepdims=True) + eps
    proj = ds - (jnp.sum(ds*dv,-1,keepdims=True)/n2)*dv  # Π(z)(ds)
    A    = proj                               # γ = -2 ⇒ |z|^0 Π(z)
    coll = (-ψx[...,None]/N * A).sum(axis=1)   # (N,2)
    return s, coll

# ── one timestep ────────────────────────────────────────────────────────
@jax.jit
def step(x, v1, v2):
    ρ   = deposit_rho(x)
    E_m = field_from_rho(ρ)
    E_p = interp_E(x, E_m)

    s, coll = score_and_collision(x, v1, v2)  # KDE & collisions
    v1 += (E_p + C_col*coll[:,0]) * dt
    v2 += (     C_col*coll[:,1]) * dt
    x  = (x + v1*dt) % L
    return x, v1, v2, E_m

#%%
# ── time loop ───────────────────────────────────────────────────────────
E_l2, t_hist = [], []
for n in trange(steps, desc="time-stepping"):
    x, v1, v2, E_m = step(x, v1, v2)
    t_hist.append((n+1)*dt)
    E_l2.append(jnp.sqrt(jnp.mean(E_m**2)))

E_l2, t_hist = jnp.array(E_l2), jnp.array(t_hist)

# ── expected decay line  γ_l + C γ_{l,c}  (small-C theory) ──────────────
γ_l   = -1/k**3 * jnp.sqrt(jnp.pi/8) * jnp.exp(-1/(2*k**2) - 1.5)
γ_lc  = -jnp.sqrt(2/(9*jnp.pi))
γ     = γ_l + C_col*γ_lc
decay = E_l2[0] * jnp.exp(γ * t_hist)

#%%
plt.semilogy(t_hist, E_l2, label="PIC + collisions")
plt.semilogy(t_hist, decay, "r--", label=rf"$\gamma = {γ:.3f}$")
plt.xlabel("t"); plt.ylabel(r"$\|E\|_{L^2}$"); plt.legend(); plt.tight_layout(); plt.show()

# %%
