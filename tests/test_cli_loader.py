"""Tests for CLI loader module."""

import pytest

from geovibes.cli.loader import res_to_ifd, RES_TO_IFD


def test_res_to_ifd_mapping():
    assert res_to_ifd(10) == 0
    assert res_to_ifd(20) == 1
    assert res_to_ifd(40) == 2
    assert res_to_ifd(80) == 3


def test_res_to_ifd_invalid():
    with pytest.raises(ValueError, match="Unsupported resolution"):
        res_to_ifd(50)
