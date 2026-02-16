#%%
# Landau-damping PIC (1d-x / 2d-v) — NumPy
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

# --- parameters ---------------------------------------------------------
N          = 100_000                        # particles
M          = 256                            # mesh cells
L          = 4 * np.pi                      # domain length (k = ½ ⇒ period 4π)
k, α       = 0.5, 1e-1
dt, T      = 0.01, 30.0
steps      = int(T / dt)
dx, η      = L / M, L / M / M * M           # η = dx (hat kernel width)
w_p        = L / N                          # particle charge ⇔ ∫ρ dx = L

# --- initial particles --------------------------------------------------
rng = np.random.default_rng(0)
def sample_ic(rng):
    # rejection sampling in one shot (acceptance ≥ 0.9)
    x_cand = rng.uniform(0., L, int(N*1.2))
    accept = rng.uniform(0., 1., x_cand.shape) < (1 + α*np.cos(k * x_cand)) / (1 + α)
    x = x_cand[accept][:N]                                   # (N,)
    v1 = rng.normal(0., 1., N)                               # Maxwellian
    v2 = rng.normal(0., 1., N)
    return x, v1, v2
x, v1, v2 = sample_ic(rng)

# --- PIC primitives -----------------------------------------------------
def deposit_rho(x):
    idx = x / dx
    i0  = np.floor(idx).astype(np.int32) % M
    f   = idx - np.floor(idx)
    i1  = (i0 + 1) % M
    w0, w1 = 1 - f, f
    ρ = np.zeros(M)
    np.add.at(ρ, i0, w_p*w0)
    np.add.at(ρ, i1, w_p*w1)
    ρ = ρ / dx           # density
    return ρ

def field_from_rho(ρ):
    δρ = ρ - ρ.mean()
    E  = np.cumsum(δρ) * dx                 # ∂xE = δρ
    return E - E.mean()                      # enforce  ⟨E⟩=0

def interp_E(x, E):
    idx = x / dx
    i0  = np.floor(idx).astype(np.int32) % M
    f   = idx - np.floor(idx)
    i1  = (i0 + 1) % M
    w0, w1 = 1 - f, f
    return w0 * E[i0] + w1 * E[i1]

def step(x, v1, v2):
    ρ   = deposit_rho(x)
    E_m = field_from_rho(ρ)
    E_p = interp_E(x, E_m)
    v1  = v1 + E_p * dt
    x   = (x + v1 * dt) % L
    return x, v1, v2, E_m

#%%
# --- time loop ----------------------------------------------------------
E_l2_hist, t_hist = [], []
for n in trange(steps, desc="Time-stepping"):
    x, v1, v2, E_m = step(x, v1, v2)
    if n % 10 == 0:                          # store every 10 steps
        t_hist.append(n*dt)
        E_l2_hist.append(np.sqrt(np.mean(E_m**2)))

E_l2_hist = np.array(E_l2_hist)
t_hist    = np.array(t_hist)

# --- linear-theory decay line -------------------------------------------
γ_l = -1/k**3 * np.sqrt(np.pi/8) * np.exp(-1/(2*k**2) - 1.5)
decay = E_l2_hist[0] * np.exp(γ_l * t_hist)

# --- plot ---------------------------------------------------------------
plt.semilogy(t_hist, E_l2_hist, label="PIC")
plt.semilogy(t_hist, decay, "r--", label="lin. theory")
plt.xlabel("t"); plt.ylabel(r"$\|E\|_{L^2}$"); plt.legend()
plt.tight_layout(); plt.show()

# %%