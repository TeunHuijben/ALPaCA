#!/usr/bin/env python

import pytest
from utils.epsilon import eps

def test_refractive_indices():
    eps_gold = eps.epsAu(600)
    eps_silver = eps.epsAg(600)
    eps_polystyrene = eps.nPSL(600)
    assert eps_gold == pytest.approx(-9.071653758599483+1.4080725191103145j, 0.01)
    assert eps_silver == pytest.approx(-14.08521276658323+0.6383016270558786j, 0.01)
    assert eps_polystyrene == pytest.approx(1.5904035251992983, 0.01)