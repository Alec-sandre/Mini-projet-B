"""
Module : integration_rectangles.py
MGA802 - Mini-Projet B

Intégration numérique par la méthode des rectangles (point milieu).
Deux implémentations :
  - Python de base (boucle for)
  - NumPy (vectorisée)

Fonction intégrée : f(x) = p1 + p2*x + p3*x^2 + p4*x^3
Intervalle        : [a, b]
Nombre de segments: n
"""

import timeit
import numpy as np


# ---------------------------------------------------------------------------
# Paramètres du problème (modifiables selon l'énoncé)
# ---------------------------------------------------------------------------

P1 = 1.0
P2 = 2.0
P3 = -3.0
P4 = 0.5

A = -2.0
B = 3.0


# ---------------------------------------------------------------------------
# Fonction à intégrer
# ---------------------------------------------------------------------------

def f(x):
    """Polynôme du 3e ordre : f(x) = p1 + p2*x + p3*x^2 + p4*x^3."""
    return P1 + P2 * x + P3 * x**2 + P4 * x**3


# ---------------------------------------------------------------------------
# Solution analytique exacte
# ---------------------------------------------------------------------------

def solution_analytique(a, b):
    """
    Calcule l'intégrale exacte de f sur [a, b] par la primitive.
    F(x) = p1*x + p2/2*x^2 + p3/3*x^3 + p4/4*x^4
    Retourne I_exact = F(b) - F(a).
    """
    def F(x):
        return P1 * x + (P2 / 2) * x**2 + (P3 / 3) * x**3 + (P4 / 4) * x**4

    return F(b) - F(a)


# ---------------------------------------------------------------------------
# Méthode des rectangles — Python de base
# ---------------------------------------------------------------------------

def integrale_rectangles_python(a, b, n):
    """
    Intégration par la méthode des rectangles (point milieu).
    Implémentation en Python pur (boucle for).

    Paramètres
    ----------
    a : float  — borne inférieure
    b : float  — borne supérieure
    n : int    — nombre de segments

    Retourne
    --------
    float — approximation de l'intégrale
    """
    h = (b - a) / n          # largeur d'un segment
    somme = 0.0

    for i in range(n):
        x_milieu = a + (i + 0.5) * h   # centre du segment i
        somme += f(x_milieu)

    return h * somme


def erreur_python(a, b, n):
    """
    Calcule l'erreur absolue entre la méthode des rectangles (Python)
    et la solution analytique.

    Retourne
    --------
    float — |I_rect - I_exact|
    """
    i_exact = solution_analytique(a, b)
    i_rect  = integrale_rectangles_python(a, b, n)
    return abs(i_rect - i_exact)


# ---------------------------------------------------------------------------
# Méthode des rectangles — NumPy (vectorisée)
# ---------------------------------------------------------------------------

def integrale_rectangles_numpy(a, b, n):
    """
    Intégration par la méthode des rectangles (point milieu).
    Implémentation vectorisée avec NumPy (pas de boucle Python).

    Paramètres
    ----------
    a : float  — borne inférieure
    b : float  — borne supérieure
    n : int    — nombre de segments

    Retourne
    --------
    float — approximation de l'intégrale
    """
    h = (b - a) / n

    # Vecteur de tous les centres en une seule opération
    i_vecteur  = np.arange(n)               # [0, 1, 2, ..., n-1]
    x_milieux  = a + (i_vecteur + 0.5) * h  # centres vectorisés

    # Évaluation de f sur tous les points simultanément (f est compatible NumPy
    # car elle n'utilise que des opérations élémentaires)
    valeurs = P1 + P2 * x_milieux + P3 * x_milieux**2 + P4 * x_milieux**3

    return h * np.sum(valeurs)


def erreur_numpy(a, b, n):
    """
    Calcule l'erreur absolue entre la méthode des rectangles (NumPy)
    et la solution analytique.

    Retourne
    --------
    float — |I_rect_numpy - I_exact|
    """
    i_exact = solution_analytique(a, b)
    i_rect  = integrale_rectangles_numpy(a, b, n)
    return abs(i_rect - i_exact)


# ---------------------------------------------------------------------------
# Analyse de convergence
# ---------------------------------------------------------------------------

def convergence_rectangles(a, b, liste_n):
    """
    Retourne les erreurs pour chaque valeur de n dans liste_n.
    Utile pour tracer la courbe de convergence.

    Retourne
    --------
    dict avec clés 'n', 'erreur_python', 'erreur_numpy'
    """
    resultats = {"n": [], "erreur_python": [], "erreur_numpy": []}

    for n in liste_n:
        resultats["n"].append(n)
        resultats["erreur_python"].append(erreur_python(a, b, n))
        resultats["erreur_numpy"].append(erreur_numpy(a, b, n))

    return resultats


# ---------------------------------------------------------------------------
# Mesure du temps d'exécution
# ---------------------------------------------------------------------------

def mesurer_temps(a, b, n, nb_repetitions=100):
    """
    Mesure le temps d'exécution des deux implémentations avec timeit.

    Retourne
    --------
    tuple (temps_python, temps_numpy) en secondes (moyenne sur nb_repetitions)
    """
    temps_python = timeit.timeit(
        lambda: integrale_rectangles_python(a, b, n),
        number=nb_repetitions
    ) / nb_repetitions

    temps_numpy = timeit.timeit(
        lambda: integrale_rectangles_numpy(a, b, n),
        number=nb_repetitions
    ) / nb_repetitions

    return temps_python, temps_numpy


# ---------------------------------------------------------------------------
# Démonstration rapide (appelée depuis main.py)
# ---------------------------------------------------------------------------

def demo_rectangles():
    """Affiche les résultats de la méthode des rectangles pour n = 10."""
    n = 10
    i_exact  = solution_analytique(A, B)
    i_python = integrale_rectangles_python(A, B, n)
    i_numpy  = integrale_rectangles_numpy(A, B, n)

    print("=" * 55)
    print("  Méthode des rectangles (point milieu)")
    print("=" * 55)
    print(f"  Paramètres : p1={P1}, p2={P2}, p3={P3}, p4={P4}")
    print(f"  Intervalle : [{A}, {B}]   |   n = {n} segments")
    print("-" * 55)
    print(f"  I exact      = {i_exact:.10f}")
    print(f"  I Python     = {i_python:.10f}   erreur = {abs(i_python - i_exact):.2e}")
    print(f"  I NumPy      = {i_numpy:.10f}   erreur = {abs(i_numpy  - i_exact):.2e}")
    print("-" * 55)

    t_py, t_np = mesurer_temps(A, B, n=10_000, nb_repetitions=50)
    print(f"  Temps Python (n=10 000) : {t_py*1000:.4f} ms")
    print(f"  Temps NumPy  (n=10 000) : {t_np*1000:.4f} ms")
    print(f"  Accélération NumPy      : x{t_py/t_np:.1f}")
    print("=" * 55)
