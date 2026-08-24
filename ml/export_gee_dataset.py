"""Export Sentinel/Landsat drought-segmentation patches from Google Earth Engine.

This module is intentionally not run on import. Authenticate locally with
``earthengine authenticate`` before invoking the CLI or FastAPI export endpoint.
The output is educational research data only and is not a validated drought product.
"""

from __future__ import annotations

import random
import tempfile
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling

from app.config import (
    DATASET_ROOT,
    GEE_GROWING_SEASON_END_MONTH,
    GEE_GROWING_SEASON_START_MONTH,
    PATCH_CHANNELS,
    PATCH_SCALE_METERS,
    PATCH_SIZE,
    VCI_DROUGHT_THRESHOLD_PERCENT,
)


# Approximate analysis windows for educational sampling, not administrative boundaries.
KAZAKHSTAN_REGIONS = {
    "akmola": [68.0, 50.5, 72.5, 54.5],
    "karaganda": [68.0, 45.0, 76.5, 50.8],
}
SPLIT_RATIOS = (("train", 0.70), ("val", 0.15), ("test", 0.15))
LANDSAT_CLEAR_PIXEL_BITS = sum(1 << bit for bit in (0, 1, 2, 3, 4, 5))


class EarthEngineConfigurationError(RuntimeError):
    """Raised when Earth Engine is unavailable or has not been authenticated."""


@dataclass(frozen=True)
class ExportResult:
    exported_patches: int
    dataset_root: Path


def _earth_engine():
    try:
        import ee
    except ImportError as error:  # pragma: no cover - dependency-level failure
        raise EarthEngineConfigurationError("earthengine-api is not installed.") from error
    return ee


def initialize_earth_engine() -> None:
    """Initialize a previously authenticated Earth Engine client without prompting.

    The server must use an existing Earth Engine credential or application-default
    credentials. This function never starts an interactive login flow.
    """
    ee = _earth_engine()
    try:
        # Earth Engine now requires a Cloud project for most API requests.
        # `earthengine set_project` also persists a default project, so this
        # explicit value is optional when the CLI has already configured one.
        project = os.getenv("GEE_PROJECT")
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as error:  # Earth Engine raises multiple implementation-specific errors.
        raise EarthEngineConfigurationError(
            "Google Earth Engine is not initialized. Run `earthengine authenticate` "
            "and `earthengine set_project YOUR_EARTH_ENGINE_PROJECT`, or set "
            "GEE_PROJECT in backend/.env."
        ) from error


def _add_ndvi(image):
    """Add a normalized-difference vegetation band to a harmonized image."""
    image = _earth_engine().Image(image)
    return image.addBands(image.normalizedDifference(["NIR", "Red"]).rename("NDVI"))


def mask_sentinel2(image):
    """Mask cloud/shadow/snow classes with Sentinel-2 L2A SCL."""
    image = _earth_engine().Image(image)
    scl = image.select("SCL")
    valid_surface = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
    reflectance = image.select(
        ["B2", "B3", "B4", "B8", "B11", "B12"],
        ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
    ).multiply(0.0001)
    return _add_ndvi(
        reflectance.updateMask(valid_surface).copyProperties(image, image.propertyNames())
    ).toFloat()


def mask_landsat(image):
    """Mask Landsat Collection 2 Level-2 cloud, shadow, snow, and fill pixels."""
    image = _earth_engine().Image(image)
    qa_pixel = image.select("QA_PIXEL")
    valid_surface = qa_pixel.bitwiseAnd(LANDSAT_CLEAR_PIXEL_BITS).eq(0)
    reflectance = image.select(
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"],
    ).multiply(0.0000275).add(-0.2)
    return _add_ndvi(
        reflectance.updateMask(valid_surface).copyProperties(image, image.propertyNames())
    ).toFloat()


def build_composite(region, year: int):
    """Return seven-channel growing-season composite and VCI drought label image."""
    ee = _earth_engine()
    start = ee.Date.fromYMD(year, GEE_GROWING_SEASON_START_MONTH, 1)
    end = ee.Date.fromYMD(year, GEE_GROWING_SEASON_END_MONTH, 31).advance(1, "day")
    sentinel = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start, end)
        .map(mask_sentinel2)
    )
    landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
        .filterBounds(region)
        .filterDate(start, end)
        .map(mask_landsat)
    )
    observations = sentinel.merge(landsat).select(list(PATCH_CHANNELS))
    composite = observations.median().select(list(PATCH_CHANNELS))
    ndvi_min = observations.select("NDVI").min()
    ndvi_max = observations.select("NDVI").max()
    ndvi_range = ndvi_max.subtract(ndvi_min).max(0.0001)
    vci = composite.select("NDVI").subtract(ndvi_min).divide(ndvi_range).multiply(100).rename("VCI")
    drought_mask = vci.lt(VCI_DROUGHT_THRESHOLD_PERCENT).rename("drought_mask").toUint8()
    return composite, drought_mask


def _grid_geometries(region, max_patches: int) -> Iterable:
    """Create deterministic, approximately 256 x 256 pixel grid cells in EPSG:3857."""
    ee = _earth_engine()
    patch_width_meters = PATCH_SIZE * PATCH_SCALE_METERS
    grid = region.coveringGrid(ee.Projection("EPSG:3857"), patch_width_meters).randomColumn("sample", 42).sort("sample")
    features = grid.toList(max_patches).getInfo()
    for feature in features:
        yield ee.Geometry(feature["geometry"])


def _read_raster(path: Path, bands: int, resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.count != bands:
            raise ValueError(f"Expected {bands} bands in {path.name}, received {source.count}.")
        return source.read(
            out_shape=(bands, PATCH_SIZE, PATCH_SIZE),
            resampling=resampling,
        )


def _download_geotiff(image, path: Path, geometry) -> None:
    """Download one Earth Engine image and raise with the API's useful error text."""
    url = image.getDownloadURL(
        {
            "name": path.stem,
            "scale": PATCH_SCALE_METERS,
            "region": geometry,
            "crs": "EPSG:3857",
            "filePerBand": False,
            "format": "GEO_TIFF",
        }
    )
    response = requests.get(url, timeout=300)
    if response.status_code != 200:
        try:
            detail = response.json()["error"]["message"]
        except (KeyError, ValueError, TypeError):
            detail = response.text[:500] or response.reason
        raise RuntimeError(f"Earth Engine tile download failed ({response.status_code}): {detail}")
    path.write_bytes(response.content)


def _download_patch(data_image, label_image, geometry, temp_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download a single remote tile as GeoTIFFs and convert it to numpy arrays."""
    patch_path = temp_dir / "patch.tif"
    stacked_image = data_image.addBands(label_image.toFloat())
    _download_geotiff(stacked_image, patch_path, geometry)
    patch = _read_raster(patch_path, len(PATCH_CHANNELS) + 1, Resampling.bilinear)
    image = patch[: len(PATCH_CHANNELS)].astype(np.float32)
    label = patch[len(PATCH_CHANNELS)].astype(np.uint8)
    return np.nan_to_num(image, nan=0.0), (label > 0).astype(np.uint8)


def _split_name(random_source: random.Random) -> str:
    draw = random_source.random()
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS:
        cumulative += ratio
        if draw < cumulative:
            return name
    return "test"


def _write_patch(image: np.ndarray, label: np.ndarray, split: str, patch_id: str, dataset_root: Path) -> None:
    image_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(image_dir / f"{patch_id}.npy", image)
    np.save(label_dir / f"{patch_id}.npy", label)


def export_dataset(regions: list[str], year: int, max_patches: int, dataset_root: Path = DATASET_ROOT) -> ExportResult:
    """Export GEE patches to train/val/test folders using a deterministic 70/15/15 split."""
    initialize_earth_engine()
    ee = _earth_engine()
    rng = random.Random(42)
    exported = 0
    remaining = max_patches
    per_region_limit = max(1, (max_patches + len(regions) - 1) // len(regions))

    for region_name in regions:
        if region_name not in KAZAKHSTAN_REGIONS:
            raise ValueError(f"Unknown region: {region_name}")
        region = ee.Geometry.Rectangle(KAZAKHSTAN_REGIONS[region_name])
        data_image, label_image = build_composite(region, year)
        for tile_number, geometry in enumerate(_grid_geometries(region, min(per_region_limit, remaining))):
            with tempfile.TemporaryDirectory(prefix="gee_patch_") as temporary:
                image, label = _download_patch(data_image, label_image, geometry, Path(temporary))
            if not np.isfinite(image).any() or image.max() == image.min():
                continue
            patch_id = f"{region_name}_{year}_{tile_number:05d}"
            _write_patch(image, label, _split_name(rng), patch_id, dataset_root)
            exported += 1
            remaining -= 1
            if remaining <= 0:
                return ExportResult(exported_patches=exported, dataset_root=dataset_root)
    return ExportResult(exported_patches=exported, dataset_root=dataset_root)


def fetch_patch_for_coordinates(latitude: float, longitude: float, year: int) -> np.ndarray:
    """Build and download a 256x256 seven-band patch centred on supplied coordinates."""
    initialize_earth_engine()
    ee = _earth_engine()
    point = ee.Geometry.Point([longitude, latitude])
    tile = point.buffer((PATCH_SIZE * PATCH_SCALE_METERS) / 2).bounds()
    data_image, label_image = build_composite(tile, year)
    with tempfile.TemporaryDirectory(prefix="gee_coordinate_") as temporary:
        image, _ = _download_patch(data_image, label_image, tile, Path(temporary))
    return image
