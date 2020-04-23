"""
File for storing concentration inequalities that rely on mean and variance.
All functions in this file take mean and variance as its first and second arguments respectively.
Returns:
    [type]: [description]
"""
from enum import Enum
tol = 1e-6

class ConcentrationInequality(Enum):
    CANTELLI = 1
    VP = 2
    GAUSS = 3

def cantelli(mean, variance):
    assert mean >= -tol
    return variance/(variance + mean**2)

def vp(mean, variance):
    assert mean >= ((5.0/3.0) * variance)**0.5 - tol
    return (4.0/9.0) * cantelli(mean, variance)

def gauss(mean, variance):
    assert mean >= (2.0) * (variance)**0.5 - tol
    return (2.0/9.0) * variance/(mean**2.0)