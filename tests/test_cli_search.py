"""Tests for CLI search module."""

import numpy as np

from geovibes.cli.search import compute_query_vector


def test_query_vector_single_positive():
    pos = [np.array([1.0, 0.0, 0.0])]
    result = compute_query_vector(pos)
    np.testing.assert_array_almost_equal(result, [2.0, 0.0, 0.0])


def test_query_vector_pos_and_neg():
    pos = [np.array([1.0, 0.0, 0.0])]
    neg = [np.array([0.0, 1.0, 0.0])]
    result = compute_query_vector(pos, neg)
    np.testing.assert_array_almost_equal(result, [2.0, -1.0, 0.0])


def test_query_vector_multiple_pos():
    pos = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    result = compute_query_vector(pos)
    np.testing.assert_array_almost_equal(result, [1.0, 1.0])


def test_query_vector_empty_returns_none():
    result = compute_query_vector([])
    assert result is None


def test_query_vector_formula():
    pos = [np.array([0.5, 0.3])]
    neg = [np.array([0.1, 0.8])]
    result = compute_query_vector(pos, neg)
    expected = 2 * np.array([0.5, 0.3]) - np.array([0.1, 0.8])
    np.testing.assert_array_almost_equal(result, expected)
