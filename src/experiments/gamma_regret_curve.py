import numpy as np
import matplotlib.pyplot as plt

from simulation import simulate

# ---------------------------------------
# Problem setup
# ---------------------------------------
k = 10
m = 3
T = 20000

means = np.linspace(0.3, 0.7, k).tolist()

c = 5
omega_max = 5

# Theoretical choices
gamma_theoretical_1 = (c + omega_max) ** 2 * m      # theoretical choice #1 
gamma_theoretical_2 = (c + omega_max) * m            # theoretical choice #2

# Base linspace over some range (adjust num / endpoints if you like)
gamma_min = 1.0
gamma_max = 1000.0
base_gammas = np.linspace(gamma_min, gamma_max, num=25).tolist()

# Combine linspace with the two theoretical values, then sort & deduplicate
gammas = sorted(set([1,2,3,5,10,15] + base_gammas + [gamma_theoretical_1, gamma_theoretical_2]))

n_runs = 30
base_seed = 12345

# ---------------------------------------
# Estimate final cumulative regret for each gamma
# ---------------------------------------
avg_final_regrets = []
std_final_regrets = []

for idx, gamma in enumerate(gammas):
    final_regrets = []

    for r in range(n_runs):
        seed = base_seed + 1000 * idx + r

        # simulate for one run and grab only final cum_regret
        _, results = simulate(
                policies=['optimistic-hire'],
                k=k,
                m=m,
                T=T,
                c=c,
                gamma=gamma,
                omega_max=omega_max,
                means=means,
                seed0=seed,
                n_runs=1
            )
        
        mean_curve, std_curve = results['optimistic-hire']
        cum_regret = mean_curve[-1]
        final_regrets.append(cum_regret)

    final_regrets = np.asarray(final_regrets, dtype=float)
    avg_final_regrets.append(final_regrets.mean())
    std_final_regrets.append(final_regrets.std())

gammas = np.asarray(gammas, dtype=float)
avg_final_regrets = np.asarray(avg_final_regrets)
std_final_regrets = np.asarray(std_final_regrets)

# ---------------------------------------
# Sort by gamma for a cleaner plot
# ---------------------------------------
order = np.argsort(gammas)
gammas_sorted = gammas[order]
avg_final_sorted = avg_final_regrets[order]
std_final_sorted = std_final_regrets[order]

# Helper to find indices of theoretical gammas
def find_index(arr: np.ndarray, val: float):
    idx = np.where(arr == val)[0]
    return int(idx[0]) if len(idx) > 0 else None

idx_mult = find_index(gammas_sorted, gamma_theoretical_1)
idx_add = find_index(gammas_sorted, gamma_theoretical_2)

# ---------------------------------------
# Plot: total cumulative regret vs gamma
# ---------------------------------------
plt.figure(figsize=(7, 5))

# Baseline line + error bars
plt.errorbar(
    gammas_sorted,
    avg_final_sorted,
    yerr=std_final_sorted,
    fmt="o-",
    linewidth=1,
    capsize=2,
    label="Average final cumulative regret",
)

# Highlight theoretical multiplicative choice
if idx_mult is not None:
    plt.scatter(
        gammas_sorted[idx_mult],
        avg_final_sorted[idx_mult],
        s=120,
        marker="s",
        edgecolor="black",
        linewidth=1.5,
        label = r"Theoretical Choice: $\gamma = (\omega_{\max} + c)^2 m$"

    )

# Highlight theoretical additive choice
if idx_add is not None:
    plt.scatter(
        gammas_sorted[idx_add],
        avg_final_sorted[idx_add],
        s=120,
        marker="D",
        edgecolor="black",
        linewidth=1.5,
        label=r"Theoretical Choice: $\gamma = (\omega_{\max} + c) m$"
    )

plt.xlabel(r"$\gamma$")
plt.ylabel(rf"Total cumulative regret at $T = {T}$")
plt.title(
    r"Final cumulative regret vs $\gamma$" + "\n" +
    rf"($k={k}$, $m={m}$, $c={c}$, $\omega_\max={omega_max}$, ${n_runs}$ runs per $\gamma$)"
)

# If the spread in γ is big, you might prefer log-scale:
# plt.xscale("log")

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
