"""
Module : integration_trapeze.py
MGA 802 - Mini-Projet B

Implémentation de la méthode d'intégration numérique des trapèzes
en Python de base et avec NumPy.
"""

import numpy as np
import timeit


# ─────────────────────────────────────────────
#  Définition de la fonction et solution exacte
# ─────────────────────────────────────────────

def fonction_polynomiale(x, p1, p2, p3, p4):
    """Évalue le polynôme f(x) = p1 + p2*x + p3*x^2 + p4*x^3."""
    return p1 + p2 * x + p3 * x**2 + p4 * x**3


def solution_analytique(a, b, p1, p2, p3, p4):
    """
    Calcule la solution analytique (exacte) de l'intégrale de f sur [a, b].

    Intégrale de (p1 + p2*x + p3*x^2 + p4*x^3) dx
      = p1*x + p2/2 * x^2 + p3/3 * x^3 + p4/4 * x^4  + C

    Paramètres
    ----------
    a, b : bornes d'intégration
    p1, p2, p3, p4 : coefficients du polynôme

    Retourne
    --------
    float : valeur exacte de l'intégrale
    """
    def integrale_polynome(x):
        return p1 * x + (p2 / 2) * x**2 + (p3 / 3) * x**3 + (p4 / 4) * x**4

    return integrale_polynome(b) - integrale_polynome(a)


def calcul_erreur(integrale_numerique, integrale_exacte):
    """
    Calcule l'erreur absolue entre l'intégrale numérique et la solution exacte.

    Paramètres
    ----------
    integrale_numerique : valeur approchée
    integrale_exacte : valeur de référence analytique

    Retourne
    --------
    float : erreur absolue |integrale_numerique - integrale_exacte|
    """
    return abs(integrale_numerique - integrale_exacte)


# ─────────────────────────────────────────────
#  Méthode des trapèzes — Python de base
# ─────────────────────────────────────────────

def trapeze_python(a, b, n, p1, p2, p3, p4):
    """
    Intégration numérique par la méthode des trapèzes (Python de base).

    Pour une section de segment [x_{i}, x_{i+1}] on peut approximer l’aire T sous la courbe avec :
        T = (x_{i+1} - x_{i}) * (f(x_{i}) + f(x_{i+1})) / 2

    Paramètres
    ----------
    a, b : bornes d'intégration
    n    : nombre de segments
    p1, p2, p3, p4 : coefficients du polynôme

    Retourne
    --------
    float : approximation de l'intégrale
    """
    h = (b - a) / n          # largeur d'un segment
    somme = 0.0

    for i in range(n):
        x_gauche = a + i * h
        x_droite = x_gauche + h

        somme += (h) * (
            fonction_polynomiale(x_gauche, p1, p2, p3, p4)
            + fonction_polynomiale(x_droite,  p1, p2, p3, p4)
        ) / 2

    return somme


def erreur_trapeze_python(n, a, b, p1, p2, p3, p4):
    """
    Retourne l'erreur absolue de la méthode des trapèzes (Python) pour n segments.

    Paramètres
    ----------
    n : nombre de segments
    a, b, p1, p2, p3, p4 : paramètres de la fonction et de l'intervalle

    Retourne
    --------
    float : erreur absolue
    """
    integrale_reference = solution_analytique(a, b, p1, p2, p3, p4)
    integrale_calculee   = trapeze_python(a, b, n, p1, p2, p3, p4)
    return calcul_erreur(integrale_calculee, integrale_reference)


# ─────────────────────────────────────────────
#  Méthode des trapèzes — NumPy vectorisé
# ─────────────────────────────────────────────

def trapeze_numpy(a, b, n, p1, p2, p3, p4):
    """
    Intégration numérique par la méthode des trapèzes (NumPy vectorisé).

    Vectorisation : tous les segments sont traités simultanément avec
    des tableaux NumPy, ce qui évite la boucle Python explicite.

    Paramètres
    ----------
    a, b : bornes d'intégration
    n    : nombre de segments
    p1, p2, p3, p4 : coefficients du polynôme

    Retourne
    --------
    float : approximation de l'intégrale
    """
    h = (b - a) / n

    # Bornes gauches et droites de chaque segment
    x_gauche = np.linspace(a, b - h,num= n)     # shape (n,)
    x_droite = x_gauche + h                     # shape (n,)

    # Évaluation vectorisée du polynôme
    f_g = p1 + p2 * x_gauche + p3 * x_gauche**2 + p4 * x_gauche**3
    f_d = p1 + p2 * x_droite + p3 * x_droite**2 + p4 * x_droite**3

    # Somme vectorisée
    return np.sum(h * (f_g + f_d) / 2)


def erreur_trapeze_numpy(n, a, b, p1, p2, p3, p4):
    """
    Retourne l'erreur absolue de la méthode des trapèzes (NumPy) pour n segments.

    Paramètres
    ----------
    n : nombre de segments
    a, b, p1, p2, p3, p4 : paramètres de la fonction et de l'intervalle

    Retourne
    --------
    float : erreur absolue
    """
    integrale_reference = solution_analytique(a, b, p1, p2, p3, p4)
    integrale_calculee   = trapeze_numpy(a, b, n, p1, p2, p3, p4)
    return calcul_erreur(integrale_calculee, integrale_reference)


# ─────────────────────────────────────────────
#  Mesure du temps d'exécution
# ─────────────────────────────────────────────

def mesurer_temps_trapeze(methode, n_values, a, b, p1, p2, p3, p4,
                          nb_repetitions=100):
    """
    Mesure le temps d'exécution moyen (timeit) d'une méthode des trapèzes
    pour une liste de valeurs de n.

    Paramètres
    ----------
    methode        : fonction d'intégration à tester (callable)
    n_values       : liste/array du nombre de segments à tester
    a, b           : bornes d'intégration
    p1, p2, p3, p4 : coefficients du polynôme
    nb_repetitions : nombre de répétitions pour timeit

    Retourne
    --------
    list[float] : temps moyen en secondes pour chaque n
    """
    temps = []
    for n in n_values:
        t = timeit.timeit(
            lambda: methode(a, b, n, p1, p2, p3, p4),
            number=nb_repetitions
        ) / nb_repetitions
        temps.append(t)
    return temps