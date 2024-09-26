#!/usr/bin/env python

"""Tests for `alpaca` package."""

import pytest
from utils.epsilon import eps



def test_refractive_indices():
    eps_gold = eps.epsAu(600)
    eps_silver = eps.epsAg(600)
    eps_polystyrene = eps.nPSL(600)
    assert eps_gold == pytest.approx(-9.071653758599483+1.4080725191103145j, 0.01)
    assert eps_silver == pytest.approx(-14.08521276658323+0.6383016270558786j, 0.01)
    assert eps_polystyrene == pytest.approx(1.5904035251992983, 0.01)


# @pytest.fixture
# def response():
#     """Sample pytest fixture.

#     See more at: http://doc.pytest.org/en/latest/fixture.html
#     """
#     # import requests
#     # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


# def test_content(response):
#     """Sample pytest test function with the pytest fixture as an argument."""
#     # from bs4 import BeautifulSoup
#     # assert 'GitHub' in BeautifulSoup(response.content).title.string
