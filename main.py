"""
main.py
MGA 802 - Mini-Projet B : Analyse numérique
Intégration numérique — Comparaison des méthodes

Ce fichier charge les modules d'intégration, appelle toutes les méthodes
et génère les graphiques de convergence et de temps d'exécution.

Usage :
    python main.py
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit

# ── Import des modules d'intégration ──────────────────────────────────────────
from Fonction.integration_rectangle import (
    solution_analytique, calcul_erreur,
    rectangles_python,  erreur_rect_python,
    rectangles_numpy,   erreur_rect_numpy,
    mesurer_temps_rect,
    demo_rectangles, convergence_rectangles,
    A, B,
)

from Fonction.integration_trapeze import (
    solution_analytique, calcul_erreur,
    trapeze_python, erreur_trapeze_python,
    trapeze_numpy,  erreur_trapeze_numpy,
    mesurer_temps_trapeze, demo_trapeze,
)

from Fonction.integration_simpson import (
    solution_analytique,
    calcul_erreur,
    f,
    simpson_python,  erreur_simpson_python,
    simpson_numpy,   erreur_simpson_numpy,
    simpson_scipy,   erreur_simpson_scipy,
    mesurer_temps_simpson,
)

# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

P1, P2, P3, P4 = 1.0, -2.0, 0.5, 0.3
A, B           = -2.0, 3.0
N_BASE         = 10
NB_REPS        = 200

N_VALUES = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], dtype=int)

# Palette cohérente pour toutes les figures
COULEURS = {
    'rect_py':  '#2196F3',
    'rect_np':  '#64B5F6',
    'trap_py':  '#FF9800',
    'trap_np':  '#FFB74D',
    'trap_sc':  '#E65100',
    'simp_py':  '#4CAF50',
    'simp_np':  '#81C784',
    'simp_sc':  '#1B5E20',
}
LABELS = {
    'rect_py': 'Rect. Python',
    'rect_np': 'Rect. NumPy',
    'trap_py': 'Trap. Python',
    'trap_np': 'Trap. NumPy',
    'trap_sc': 'Trap. SciPy',
    'simp_py': 'Simpson Python',
    'simp_np': 'Simpson NumPy',
    'simp_sc': 'Simpson SciPy',
}
LS = {
    'rect_py': '-',  'rect_np': '--',
    'trap_py': '-',  'trap_np': '--',  'trap_sc': ':',
    'simp_py': '-',  'simp_np': '--',  'simp_sc': ':',
}
MK = {
    'rect_py': 'o',  'rect_np': 's',
    'trap_py': '^',  'trap_np': 'D',  'trap_sc': 'x',
    'simp_py': 'o',  'simp_np': 's',  'simp_sc': '^',
}

# ══════════════════════════════════════════════════════════════════════════════
#  1. SOLUTION EXACTE
# ══════════════════════════════════════════════════════════════════════════════

I_exact = solution_analytique(A, B, P1, P2, P3, P4)
print("=" * 60)
print("MGA 802 — Mini-Projet B : Intégration numérique")
print("=" * 60)
print(f"Paramètres  : p1={P1}, p2={P2}, p3={P3}, p4={P4}")
print(f"Intervalle  : [{A}, {B}]")
print(f"I_exact     = {I_exact:.10f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  2. RÉSULTATS POUR N_BASE SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════

demo_rectangles(A, B, P1, P2, P3, P4, N_BASE)
print()

demo_trapeze(A, B, N_BASE, P1, P2, P3, P4, NB_REPS)
print()

print(f"── Méthode de Simpson  (n = {N_BASE}) ──────────────────────")
for label, fn in [("Python", simpson_python), ("NumPy", simpson_numpy), ("SciPy", simpson_scipy)]:
    I   = fn(A, B, N_BASE, P1, P2, P3, P4)
    err = calcul_erreur(I, I_exact)
    t   = timeit.timeit(lambda fn=fn: fn(A, B, N_BASE, P1, P2, P3, P4), number=NB_REPS) / NB_REPS
    print(f"  {label:<8}: I = {I:.10f}  |  erreur = {err:.2e}  |  temps = {t*1e6:.2f} µs")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  3. CALCUL DES ERREURS ET TEMPS POUR TOUTES LES MÉTHODES
# ══════════════════════════════════════════════════════════════════════════════

print("Calcul des erreurs et temps pour toutes les méthodes...")

# ── Erreurs ──────────────────────────────────────────────────────────────────
erreurs = {
    'rect_py': [erreur_rect_python(n, A, B, P1, P2, P3, P4)   for n in N_VALUES],
    'rect_np': [erreur_rect_numpy(n, A, B, P1, P2, P3, P4)    for n in N_VALUES],
    'trap_py': [erreur_trapeze_python(A, B, n, P1, P2, P3, P4) for n in N_VALUES],
    'trap_np': [erreur_trapeze_numpy(A, B, n, P1, P2, P3, P4)  for n in N_VALUES],
    'trap_sc': [calcul_erreur(
                    __import__('scipy.integrate', fromlist=['trapezoid']).trapezoid(
                        P1 + P2*np.linspace(A,B,n+1) + P3*np.linspace(A,B,n+1)**2 + P4*np.linspace(A,B,n+1)**3,
                        x=np.linspace(A,B,n+1)),
                    I_exact) for n in N_VALUES],
    'simp_py': [erreur_simpson_python(n, A, B, P1, P2, P3, P4) for n in N_VALUES],
    'simp_np': [erreur_simpson_numpy(n, A, B, P1, P2, P3, P4)  for n in N_VALUES],
    'simp_sc': [erreur_simpson_scipy(n, A, B, P1, P2, P3, P4)  for n in N_VALUES],
}

# ── Temps ─────────────────────────────────────────────────────────────────────
from scipy.integrate import trapezoid as sci_trap, simpson as sci_simp

def _trap_scipy(a, b, n, p1, p2, p3, p4):
    x = np.linspace(a, b, n+1)
    y = p1 + p2*x + p3*x**2 + p4*x**3
    return float(sci_trap(y, x=x))

temps = {
    'rect_py': mesurer_temps_rect(rectangles_python, N_VALUES, A, B, P1, P2, P3, P4, nb_repetitions=NB_REPS),
    'rect_np': mesurer_temps_rect(rectangles_numpy,  N_VALUES, A, B, P1, P2, P3, P4, nb_repetitions=NB_REPS),
    'trap_py': [timeit.timeit(lambda n=n: trapeze_python(A,B,n,P1,P2,P3,P4), number=NB_REPS)/NB_REPS for n in N_VALUES],
    'trap_np': [timeit.timeit(lambda n=n: trapeze_numpy(A,B,n,P1,P2,P3,P4),  number=NB_REPS)/NB_REPS for n in N_VALUES],
    'trap_sc': [timeit.timeit(lambda n=n: _trap_scipy(A,B,n,P1,P2,P3,P4),    number=NB_REPS)/NB_REPS for n in N_VALUES],
    'simp_py': mesurer_temps_simpson(simpson_python, N_VALUES, A, B, P1, P2, P3, P4, nb_repetitions=NB_REPS),
    'simp_np': mesurer_temps_simpson(simpson_numpy,  N_VALUES, A, B, P1, P2, P3, P4, nb_repetitions=NB_REPS),
    'simp_sc': mesurer_temps_simpson(simpson_scipy,  N_VALUES, A, B, P1, P2, P3, P4, nb_repetitions=NB_REPS),
}

print("Calcul terminé.")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  4. FIGURE 1 — CONVERGENCE (toutes méthodes)
# ══════════════════════════════════════════════════════════════════════════════

fig1, ax1 = plt.subplots(figsize=(9, 5))

for k, vals in erreurs.items():
    y = [max(v, 1e-16) for v in vals]
    ax1.loglog(N_VALUES, y, LS[k], marker=MK[k], ms=5,
               color=COULEURS[k], label=LABELS[k], lw=1.6)

ax1.set_xlabel("Nombre de segments n", fontsize=11)
ax1.set_ylabel("Erreur absolue |I_num − I_exact|", fontsize=11)
ax1.set_title("Convergence de toutes les méthodes", fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, ncol=2, loc='lower left')
ax1.grid(True, which='both', linestyle=':', alpha=0.5)
ax1.tick_params(labelsize=9)
fig1.tight_layout()
fig1.savefig("convergence_toutes_methodes.png", dpi=150)
print("Graphique sauvegardé : convergence_toutes_methodes.png")

# ══════════════════════════════════════════════════════════════════════════════
#  5. FIGURE 2 — TEMPS DE CALCUL (toutes méthodes)
# ══════════════════════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(9, 5))

for k, vals in temps.items():
    ax2.loglog(N_VALUES, [t * 1e6 for t in vals], LS[k], marker=MK[k], ms=5,
               color=COULEURS[k], label=LABELS[k], lw=1.6)

ax2.set_xlabel("Nombre de segments n", fontsize=11)
ax2.set_ylabel("Temps d'exécution moyen (µs)", fontsize=11)
ax2.set_title("Temps de calcul de toutes les méthodes", fontsize=13, fontweight='bold')
ax2.legend(fontsize=8, ncol=2, loc='upper left')
ax2.grid(True, which='both', linestyle=':', alpha=0.5)
ax2.tick_params(labelsize=9)
fig2.tight_layout()
fig2.savefig("temps_toutes_methodes.png", dpi=150)
print("Graphique sauvegardé : temps_toutes_methodes.png")

# ══════════════════════════════════════════════════════════════════════════════
#  6. FIGURE 3 — ERREUR PAR MÉTHODE (barres groupées pour n = 10, 100, 1000)
# ══════════════════════════════════════════════════════════════════════════════

n_idx = {10: 1, 100: 4, 1000: 7}
groupes = [
    ("Rectangles", ['rect_py', 'rect_np']),
    ("Trapèzes",   ['trap_py', 'trap_np', 'trap_sc']),
    ("Simpson",    ['simp_py', 'simp_np', 'simp_sc']),
]

fig3, axes3 = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
bar_colors = ['#2196F3', '#FF9800', '#4CAF50']

for ax3, (titre, keys) in zip(axes3, groupes):
    x = np.arange(len(keys))
    width = 0.22
    for ni, (n_val, idx) in enumerate(n_idx.items()):
        vals = [max(erreurs[k][idx], 1e-16) for k in keys]
        ax3.bar(x + ni * width, vals, width,
                color=bar_colors[ni], alpha=0.75,
                label=f'n = {n_val}' if ax3 == axes3[0] else '')
    ax3.set_yscale('log')
    ax3.set_title(titre, fontsize=10, fontweight='bold')
    short = ['Python', 'NumPy'] if len(keys) == 2 else ['Python', 'NumPy', 'SciPy']
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(short, fontsize=9)
    ax3.grid(True, axis='y', linestyle=':', alpha=0.5)
    ax3.tick_params(labelsize=8)

axes3[0].set_ylabel("Erreur absolue", fontsize=10)
fig3.legend(fontsize=8, loc='upper right', ncol=1)
fig3.suptitle("Erreur par méthode et implémentation", fontsize=12, fontweight='bold')
fig3.tight_layout()
fig3.savefig("erreur_par_methode.png", dpi=150)
print("Graphique sauvegardé : erreur_par_methode.png")

# ══════════════════════════════════════════════════════════════════════════════
#  7. FIGURE 4 — ILLUSTRATION DE LA MÉTHODE DE SIMPSON
# ══════════════════════════════════════════════════════════════════════════════

def tracer_simpson(a, b, n, p1, p2, p3, p4, ax, titre=""):
    """Trace la courbe et les paraboles de Simpson pour n segments."""
    h   = (b - a) / n
    x_c = np.linspace(a, b, 400)
    y_c = p1 + p2 * x_c + p3 * x_c**2 + p4 * x_c**3

    ax.plot(x_c, y_c, 'k-', linewidth=2, label='f(x)')

    for i in range(n):
        x_g = a + i * h
        x_d = x_g + h
        x_m = (x_g + x_d) / 2.0

        pts_x = np.array([x_g, x_m, x_d])
        pts_y = np.array([f(xi, p1, p2, p3, p4) for xi in pts_x])
        coeffs = np.polyfit(pts_x, pts_y, 2)
        x_par  = np.linspace(x_g, x_d, 50)
        y_par  = np.polyval(coeffs, x_par)

        couleur = '#a8d8ea' if i % 2 == 0 else '#f9c784'
        ax.fill_between(x_par, 0, y_par, alpha=0.55, color=couleur)
        ax.plot(x_par, y_par, color='gray', linewidth=0.7)

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_title(titre, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4))
for ax_i, n_i in zip(axes4, [4, 10, 40]):
    tracer_simpson(A, B, n_i, P1, P2, P3, P4, ax_i, titre=f"Simpson n = {n_i}")
fig4.suptitle("Illustration de la méthode de Simpson", fontsize=13, fontweight='bold')
fig4.tight_layout()
fig4.savefig("illustration_simpson.png", dpi=150)
print("Graphique sauvegardé : illustration_simpson.png")

# ══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE
# ══════════════════════════════════════════════════════════════════════════════

plt.show()
print()
print("Exécution terminée.")
