"""
Module : integration_rectangle.py
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
# Paramètres par défaut
# ---------------------------------------------------------------------------

A = -2.0
B =  3.0


# ---------------------------------------------------------------------------
# Fonction à intégrer
# ---------------------------------------------------------------------------

def f(x, p1, p2, p3, p4):
    """Polynôme du 3e ordre : f(x) = p1 + p2*x + p3*x^2 + p4*x^3."""
    return p1 + p2 * x + p3 * x**2 + p4 * x**3


# ---------------------------------------------------------------------------
# Solution analytique exacte
# ---------------------------------------------------------------------------

def solution_analytique(a, b, p1, p2, p3, p4):
    """
    Calcule l'intégrale exacte de f sur [a, b] par la primitive.
    F(x) = p1*x + p2/2*x^2 + p3/3*x^3 + p4/4*x^4
    Retourne I_exact = F(b) - F(a).
    """
    def F(x):
        return p1 * x + (p2 / 2) * x**2 + (p3 / 3) * x**3 + (p4 / 4) * x**4

    return F(b) - F(a)


def calcul_erreur(i_num, i_exact):
    """Retourne l'erreur absolue |I_num - I_exact|."""
    return abs(i_num - i_exact)


# ---------------------------------------------------------------------------
# Méthode des rectangles — Python de base
# ---------------------------------------------------------------------------

def rectangles_python(a, b, n, p1, p2, p3, p4):
    """
    Intégration par la méthode des rectangles (point milieu).
    Implémentation en Python pur (boucle for).
    """
    h = (b - a) / n
    somme = 0.0
    for i in range(n):
        x_milieu = a + (i + 0.5) * h
        somme += f(x_milieu, p1, p2, p3, p4)
    return h * somme


def erreur_rect_python(n, a, b, p1, p2, p3, p4):
    """Erreur absolue de la méthode des rectangles Python."""
    i_exact = solution_analytique(a, b, p1, p2, p3, p4)
    return calcul_erreur(rectangles_python(a, b, n, p1, p2, p3, p4), i_exact)


# ---------------------------------------------------------------------------
# Méthode des rectangles — NumPy (vectorisée)
# ---------------------------------------------------------------------------

def rectangles_numpy(a, b, n, p1, p2, p3, p4):
    """
    Intégration par la méthode des rectangles (point milieu).
    Implémentation vectorisée avec NumPy (pas de boucle Python).
    """
    h = (b - a) / n
    x_milieux = a + (np.arange(n) + 0.5) * h
    valeurs = p1 + p2 * x_milieux + p3 * x_milieux**2 + p4 * x_milieux**3
    return h * np.sum(valeurs)


def erreur_rect_numpy(n, a, b, p1, p2, p3, p4):
    """Erreur absolue de la méthode des rectangles NumPy."""
    i_exact = solution_analytique(a, b, p1, p2, p3, p4)
    return calcul_erreur(rectangles_numpy(a, b, n, p1, p2, p3, p4), i_exact)


# ---------------------------------------------------------------------------
# Mesure du temps d'exécution
# ---------------------------------------------------------------------------

def mesurer_temps_rect(methode, n_values, a, b, p1, p2, p3, p4,
                       nb_repetitions=200):
    """
    Mesure le temps moyen d'exécution d'une méthode pour chaque n dans n_values.

    Paramètres
    ----------
    methode        : fonction à mesurer (rectangles_python ou rectangles_numpy)
    n_values       : tableau/liste de valeurs de n
    nb_repetitions : nombre de répétitions timeit

    Retourne
    --------
    list de temps moyens en secondes
    """
    temps = []
    for n in n_values:
        t = timeit.timeit(
            lambda n=n: methode(a, b, n, p1, p2, p3, p4),
            number=nb_repetitions
        ) / nb_repetitions
        temps.append(t)
    return temps


# ---------------------------------------------------------------------------
# Démonstration rapide
# ---------------------------------------------------------------------------

def demo_rectangles(a, b, p1, p2, p3, p4, n=10):
    """Affiche les résultats de la méthode des rectangles pour n segments."""
    i_exact  = solution_analytique(a, b, p1, p2, p3, p4)
    i_python = rectangles_python(a, b, n, p1, p2, p3, p4)
    i_numpy  = rectangles_numpy(a, b, n, p1, p2, p3, p4)




# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

def convergence_rectangles(a, b, liste_n, p1, p2, p3, p4):
    """
    Retourne les erreurs pour chaque valeur de n dans liste_n.

    Retourne
    --------
    dict avec clés 'n', 'erreur_python', 'erreur_numpy'
    """
    resultats = {"n": [], "erreur_python": [], "erreur_numpy": []}
    for n in liste_n:
        resultats["n"].append(n)
        resultats["erreur_python"].append(erreur_rect_python(n, a, b, p1, p2, p3, p4))
        resultats["erreur_numpy"].append(erreur_rect_numpy(n, a, b, p1, p2, p3, p4))
    return resultats
