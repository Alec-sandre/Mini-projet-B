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
import scipy

# ── Import des modules d'intégration ──────────────────────────────────────────
# NOTE : les modules integration_rectangles.py et integration_trapezes.py
#        sont à intégrer par l'autre membre de l'équipe.
#        Décommentez les imports ci-dessous lorsqu'ils seront disponibles.
#
# from integration_rectangles import (
#     solution_analytique, calcul_erreur,
#     rectangles_python, erreur_rect_python,
#     rectangles_numpy,  erreur_rect_numpy,
#     mesurer_temps_rect,
# )
# from integration_trapezes import (
#     trapezes_python,  erreur_trap_python,
#     trapezes_numpy,   erreur_trap_numpy,
#     trapezes_scipy,   erreur_trap_scipy,
#     mesurer_temps_trap,
# )

from Fonction.integration_rectangle import (
    solution_analytique, calcul_erreur,
    rectangles_python,  erreur_rect_python,
    rectangles_numpy,   erreur_rect_numpy,
    mesurer_temps_rect,
    demo_rectangles, convergence_rectangles,
    A, B,
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
#  PARAMÈTRES — à modifier selon l'énoncé du cours
# ══════════════════════════════════════════════════════════════════════════════

P1, P2, P3, P4 = 1.0, -2.0, 0.5, 0.3   # coefficients du polynôme
A, B           = -2.0, 3.0              # bornes d'intégration
N_BASE         = 10                     # nombre de segments de référence
NB_REPS        = 200                    # répétitions timeit

# Grille de segments pour les courbes de convergence et de temps
N_VALUES = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
                    dtype=int)


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
#  2. MÉTHODE DES RECTANGLES — résultats pour N_BASE segments
# ══════════════════════════════════════════════════════════════════════════════

demo_rectangles(A, B, P1, P2, P3, P4, N_BASE)

print(f"── Méthode des rectangles  (n = {N_BASE}) ──────────────────────")

I_rect_py   = rectangles_python(A, B, N_BASE, P1, P2, P3, P4)
err_rect_py = calcul_erreur(I_rect_py, I_exact)
t_rect_py   = timeit.timeit(
    lambda: rectangles_python(A, B, N_BASE, P1, P2, P3, P4),
    number=NB_REPS
) / NB_REPS

I_rect_np   = rectangles_numpy(A, B, N_BASE, P1, P2, P3, P4)
err_rect_np = calcul_erreur(I_rect_np, I_exact)
t_rect_np   = timeit.timeit(
    lambda: rectangles_numpy(A, B, N_BASE, P1, P2, P3, P4),
    number=NB_REPS
) / NB_REPS

print(f"  Python  : I = {I_rect_py:.10f}  |  erreur = {err_rect_py:.2e}"
      f"  |  temps = {t_rect_py*1e6:.2f} µs")
print(f"  NumPy   : I = {I_rect_np:.10f}  |  erreur = {err_rect_np:.2e}"
      f"  |  temps = {t_rect_np*1e6:.2f} µs")
print()

# ══════════════════════════════════════════════════════════════════════════════
#  3. MÉTHODE DE SIMPSON — résultats pour N_BASE segments
# ══════════════════════════════════════════════════════════════════════════════

print(f"── Méthode de Simpson  (n = {N_BASE}) ──────────────────────")

# Python de base
I_simp_py  = simpson_python(A, B, N_BASE, P1, P2, P3, P4)
err_simp_py = calcul_erreur(I_simp_py, I_exact)
t_simp_py  = timeit.timeit(
    lambda: simpson_python(A, B, N_BASE, P1, P2, P3, P4),
    number=NB_REPS
) / NB_REPS

# NumPy
I_simp_np  = simpson_numpy(A, B, N_BASE, P1, P2, P3, P4)
err_simp_np = calcul_erreur(I_simp_np, I_exact)
t_simp_np  = timeit.timeit(
    lambda: simpson_numpy(A, B, N_BASE, P1, P2, P3, P4),
    number=NB_REPS
) / NB_REPS

# SciPy
I_simp_sc  = simpson_scipy(A, B, N_BASE, P1, P2, P3, P4)
err_simp_sc = calcul_erreur(I_simp_sc, I_exact)
t_simp_sc  = timeit.timeit(
    lambda: simpson_scipy(A, B, N_BASE, P1, P2, P3, P4),
    number=NB_REPS
) / NB_REPS

print(f"  Python  : I = {I_simp_py:.10f}  |  erreur = {err_simp_py:.2e}"
      f"  |  temps = {t_simp_py*1e6:.2f} µs")
print(f"  NumPy   : I = {I_simp_np:.10f}  |  erreur = {err_simp_np:.2e}"
      f"  |  temps = {t_simp_np*1e6:.2f} µs")
print(f"  SciPy   : I = {I_simp_sc:.10f}  |  erreur = {err_simp_sc:.2e}"
      f"  |  temps = {t_simp_sc*1e6:.2f} µs")
print()


# ══════════════════════════════════════════════════════════════════════════════
#  4. CONVERGENCE EN FONCTION DE n
# ══════════════════════════════════════════════════════════════════════════════

erreurs_simp_py = [erreur_simpson_python(n, A, B, P1, P2, P3, P4)
                   for n in N_VALUES]
erreurs_simp_np = [erreur_simpson_numpy(n, A, B, P1, P2, P3, P4)
                   for n in N_VALUES]
erreurs_simp_sc = [erreur_simpson_scipy(n, A, B, P1, P2, P3, P4)
                   for n in N_VALUES]

# ── Graphique convergence Simpson ────────────────────────────────────────────
fig_conv, ax_conv = plt.subplots(figsize=(8, 5))

ax_conv.loglog(N_VALUES, erreurs_simp_py, 'o-',  label='Simpson Python',
               color='steelblue')
ax_conv.loglog(N_VALUES, erreurs_simp_np, 's--', label='Simpson NumPy',
               color='darkorange')
ax_conv.loglog(N_VALUES, erreurs_simp_sc, '^:', label='Simpson SciPy',
               color='green')

# Ligne de référence pente -4 (ordre 4 attendu pour Simpson sur polynôme deg 3)
ref_x = np.array([N_VALUES[0], N_VALUES[-1]], dtype=float)
ref_y = erreurs_simp_py[0] * (ref_x / N_VALUES[0])**(-4)
ax_conv.loglog(ref_x, ref_y, 'k--', alpha=0.4, label='pente −4 (référence)')

ax_conv.set_xlabel("Nombre de segments n")
ax_conv.set_ylabel("Erreur absolue |I_num − I_exact|")
ax_conv.set_title("Convergence — Méthode de Simpson")
ax_conv.legend()
ax_conv.grid(True, which='both', linestyle=':', alpha=0.6)
fig_conv.tight_layout()
fig_conv.savefig("convergence_simpson.png", dpi=150)
print("Graphique sauvegardé : convergence_simpson.png")


# ══════════════════════════════════════════════════════════════════════════════
#  5. TEMPS D'EXÉCUTION EN FONCTION DE n
# ══════════════════════════════════════════════════════════════════════════════

temps_simp_py = mesurer_temps_simpson(simpson_python, N_VALUES,
                                      A, B, P1, P2, P3, P4,
                                      nb_repetitions=NB_REPS)
temps_simp_np = mesurer_temps_simpson(simpson_numpy,  N_VALUES,
                                      A, B, P1, P2, P3, P4,
                                      nb_repetitions=NB_REPS)
temps_simp_sc = mesurer_temps_simpson(simpson_scipy,  N_VALUES,
                                      A, B, P1, P2, P3, P4,
                                      nb_repetitions=NB_REPS)

# ── Graphique temps Simpson ───────────────────────────────────────────────────
fig_t, ax_t = plt.subplots(figsize=(8, 5))

ax_t.loglog(N_VALUES, [t * 1e6 for t in temps_simp_py], 'o-',
            label='Simpson Python', color='steelblue')
ax_t.loglog(N_VALUES, [t * 1e6 for t in temps_simp_np], 's--',
            label='Simpson NumPy',  color='darkorange')
ax_t.loglog(N_VALUES, [t * 1e6 for t in temps_simp_sc], '^:',
            label='Simpson SciPy', color='green')

ax_t.set_xlabel("Nombre de segments n")
ax_t.set_ylabel("Temps d'exécution moyen (µs)")
ax_t.set_title("Temps de calcul — Méthode de Simpson")
ax_t.legend()
ax_t.grid(True, which='both', linestyle=':', alpha=0.6)
fig_t.tight_layout()
fig_t.savefig("temps_simpson.png", dpi=150)
print("Graphique sauvegardé : temps_simpson.png")


# ══════════════════════════════════════════════════════════════════════════════
#  6. ILLUSTRATION DE LA MÉTHODE (optionnel — visualisation pédagogique)
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

        # Parabole interpolante sur 3 points
        pts_x = np.array([x_g, x_m, x_d])
        pts_y = np.array([f(xi, p1, p2, p3, p4) for xi in pts_x])
        coeffs = np.polyfit(pts_x, pts_y, 2)
        x_par  = np.linspace(x_g, x_d, 50)
        y_par  = np.polyval(coeffs, x_par)

        # Remplissage alterné pour lisibilité
        couleur = '#a8d8ea' if i % 2 == 0 else '#f9c784'
        ax.fill_between(x_par, 0, y_par, alpha=0.55, color=couleur)
        ax.plot(x_par, y_par, color='gray', linewidth=0.7)

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_title(titre)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


fig_ill, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax_i, n_i in zip(axes, [4, 10, 40]):
    tracer_simpson(A, B, n_i, P1, P2, P3, P4, ax_i,
                   titre=f"Simpson n = {n_i}")
fig_ill.suptitle("Illustration de la méthode de Simpson", fontsize=13)
fig_ill.tight_layout()
fig_ill.savefig("illustration_simpson.png", dpi=150)
print("Graphique sauvegardé : illustration_simpson.png")

plt.show()

print()
print("Exécution terminée.")