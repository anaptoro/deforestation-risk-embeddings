from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import ee
def aef_ic() -> ee.ImageCollection:
    return ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

def modis_lc_ic() -> ee.ImageCollection:
    return ee.ImageCollection("MODIS/061/MCD12Q1")

def roads_br_fc() -> ee.FeatureCollection:
    return ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/Central-South-America")


def aef_for_year(year: int, region: ee.Geometry) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return aef_ic().filterDate(start, end).filterBounds(region).mosaic().clip(region)


def modis_lc_for_year(year: int, region: ee.Geometry) -> ee.Image:
    # MODIS MCD12Q1 has one image per year; filter by calendar year.
    img = (
        modis_lc_ic().filter(ee.Filter.calendarRange(year, year, "year"))
        .first()
        .select("LC_Type1")
        .clip(region)
    )
    return img


def is_forest_igbp(lc_type1: ee.Image) -> ee.Image:
    """Strict IGBP forest: classes 1..5."""
    return (
        lc_type1.eq(1)
        .Or(lc_type1.eq(2))
        .Or(lc_type1.eq(3))
        .Or(lc_type1.eq(4))
        .Or(lc_type1.eq(5))
    )


def forest_mask_for_year(t_year: int, region: ee.Geometry) -> ee.Image:
    lc_t = modis_lc_for_year(t_year, region)
    forest = is_forest_igbp(lc_t).rename("forest").toByte()
    return forest.updateMask(lc_t.mask())


def frontier_features_for_year(
    t_year: int,
    region: ee.Geometry,
    scale: int,
    road_search_radius_m: int = 100_000,
    nf_max_km: Optional[float] = None,
) -> ee.Image:
    lc_t = modis_lc_for_year(t_year, region)
    forest = is_forest_igbp(lc_t).rename("forest").toByte()
    nonforest = forest.Not().rename("nonforest").toByte()

    pix_m = ee.Number(lc_t.projection().nominalScale())

    dist_nf = (
        nonforest.selfMask()
        .fastDistanceTransform(256)
        .sqrt()
        .multiply(pix_m)
        .rename("dist_to_nonforest_m")
        .updateMask(forest)
    )

    if nf_max_km is not None:
        dist_nf = dist_nf.clamp(0, ee.Number(nf_max_km).multiply(1000))

    dist_rd = (
        roads_br_fc().filterBounds(region)
        .distance(searchRadius=road_search_radius_m)
        .rename("dist_to_road_m")
        .clip(region)
    )

    return dist_nf.addBands(dist_rd)


def label_basic_loss(t_year: int, region: ee.Geometry) -> ee.Image:
    # pos = forest(t)=1 & forest(t+1)=0
    lc_t = modis_lc_for_year(t_year, region)
    lc_t1 = modis_lc_for_year(t_year + 1, region)

    forest_t = is_forest_igbp(lc_t)
    forest_t1 = is_forest_igbp(lc_t1)

    y = forest_t.And(forest_t1.Not()).rename("label").toByte()
    valid = lc_t.mask().And(lc_t1.mask())
    return y.updateMask(valid)


def label_stable_loss(t_year: int, region: ee.Geometry) -> ee.Image:
    # pos = forest(t-1)=1 & forest(t)=1 & nonforest(t+1)=1
    # neg = forest(t-1)=1 & forest(t)=1 & forest(t+1)=1
    lc_tm1 = modis_lc_for_year(t_year - 1, region)
    lc_t = modis_lc_for_year(t_year, region)
    lc_tp1 = modis_lc_for_year(t_year + 1, region)

    f_tm1 = is_forest_igbp(lc_tm1)
    f_t = is_forest_igbp(lc_t)
    f_tp1 = is_forest_igbp(lc_tp1)

    pos = f_tm1.And(f_t).And(f_tp1.Not())
    neg = f_tm1.And(f_t).And(f_tp1)

    y = pos.rename("label").toByte()
    y = y.where(neg, 0)

    keep = pos.Or(neg)
    valid = lc_tm1.mask().And(lc_t.mask()).And(lc_tp1.mask())
    return y.updateMask(keep).updateMask(valid)


def label_for_year(t_year: int, region: ee.Geometry, use_stable_label: bool) -> ee.Image:
    return label_stable_loss(t_year, region) if use_stable_label else label_basic_loss(t_year, region)


def sampling_image_for_year(
    t_year: int,
    region: ee.Geometry,
    scale: int,
    use_stable_label: bool,
    road_search_radius_m: int = 100_000,
    nf_max_km: Optional[float] = None,
) -> ee.Image:
    X = aef_for_year(t_year, region)  # A00..A63
    F = frontier_features_for_year(
        t_year, region, scale, road_search_radius_m=road_search_radius_m, nf_max_km=nf_max_km
    )
    y = label_for_year(t_year, region, use_stable_label)
    return X.addBands(F).addBands(y)


def stratified_samples_for_year(
    t_year: int,
    region: ee.Geometry,
    n_neg: int,
    n_pos: int,
    scale: int,
    seed: int,
    use_stable_label: bool,
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    img = sampling_image_for_year(t_year, region, scale, use_stable_label)

    fc = img.stratifiedSample(
        numPoints=1,
        classBand="label",
        classValues=[0, 1],
        classPoints=[n_neg, n_pos],
        region=region,
        scale=scale,
        seed=seed + t_year,
        dropNulls=True,
        tileScale=tile_scale,
        geometries=True,
    )
    return fc.map(lambda f: f.set({"tYear": t_year}))


def unbiased_forest_samples(
    t_year: int,
    region: ee.Geometry,
    n_pixels: int,
    scale: int,
    seed: int,
    use_stable_label: bool,
    tile_scale: int = 4,
) -> ee.FeatureCollection:
    img = sampling_image_for_year(t_year, region, scale, use_stable_label)
    forest = forest_mask_for_year(t_year, region)
    img_forest = img.updateMask(forest)

    fc = (
        img_forest.sample(
            region=region,
            scale=scale,
            numPixels=n_pixels,
            seed=seed + 999,
            tileScale=tile_scale,
            geometries=True,
        )
        .map(lambda f: f.set({"tYear": t_year, "unbiased": 1}))
    )
    return fc


def export_fc_to_drive(
    fc: ee.FeatureCollection,
    description: str,
    filename_prefix: str,
    selectors: List[str],
    folder: Optional[str] = None,
) -> ee.batch.Task:
    kwargs = dict(
        collection=fc.select(selectors),
        description=description,
        fileNamePrefix=filename_prefix,
        fileFormat="CSV",
    )
    if folder:
        kwargs["folder"] = folder

    task = ee.batch.Export.table.toDrive(**kwargs)
    task.start()
    return task
