"""CHIRPS sampling tests on a synthetic GeoTIFF (no network)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import chirps  # noqa: E402

rasterio = pytest.importorskip("rasterio")


@pytest.fixture
def tif(tmp_path):
    from rasterio.transform import from_origin
    a = np.arange(100, dtype="float32").reshape(10, 10)
    a[0, 0] = -9999          # nodata
    a[5, 5] = -1             # negative -> treated as missing
    p = tmp_path / "t.tif"
    with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(70, 30, 0.05, 0.05), nodata=-9999) as d:
        d.write(a, 1)
    return p


def test_nearest_and_3x3(tif):
    # (row 0, col 1): edge row; neighbours nodata,1,2 / 10,11,12 -> mean of 5 valid
    # (row 2, col 2): interior, all 9 valid
    # (row 5, col 4): neighbour (5,5) is negative -> excluded
    w = pd.DataFrame({"lat": [29.975, 29.875, 29.725], "lon": [70.075, 70.125, 70.225]})
    near, m3, nv = chirps.sample_nearest_and_3x3(tif, w)
    assert near.tolist() == [1.0, 22.0, 54.0]
    assert np.isclose(m3[0], (1 + 2 + 10 + 11 + 12) / 5) and nv[0] == 5
    assert np.isclose(m3[1], 22.0) and nv[1] == 9
    vals = [43, 44, 45, 53, 54, 63, 64, 65]
    assert np.isclose(m3[2], np.mean(vals)) and nv[2] == 8
    np.testing.assert_array_equal(near, chirps.sample(tif, w))   # same nearest rule as stage 4
