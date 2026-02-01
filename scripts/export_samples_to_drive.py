from __future__ import annotations

import argparse
from typing import List

import ee

from src.gee.sampling import (
    aef_for_year,
    sampling_image_for_year,
    stratified_samples_for_year,
    unbiased_forest_samples,
    export_fc_to_drive,
)


def parse_bbox(s: str) -> ee.Geometry:
    # "xmin,ymin,xmax,ymax"
    xmin, ymin, xmax, ymax = [float(x.strip()) for x in s.split(",")]
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])


def main():
    ap = argparse.ArgumentParser(description="Export training/unbiased CSV tables from Earth Engine to Google Drive.")
    ap.add_argument("--bbox", required=True, help="xmin,ymin,xmax,ymax (lon/lat)")
    ap.add_argument("--scale", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_years", default="2018,2019,2020")
    ap.add_argument("--n_pos", type=int, default=5000)
    ap.add_argument("--n_neg", type=int, default=5000)

    ap.add_argument("--unbiased_year", type=int, default=2022)
    ap.add_argument("--n_unbiased", type=int, default=30000)

    ap.add_argument("--use_stable_label", action="store_true")
    ap.add_argument("--drive_folder", default=None, help="Optional Drive folder name")
    ap.add_argument("--prefix", default="defrisk_v1", help="Filename prefix for exports")

    args = ap.parse_args()

    # Auth/init
    ee.Initialize()

    roi = parse_bbox(args.bbox)
    train_years: List[int] = [int(x) for x in args.train_years.split(",") if x.strip()]

    # Determine band list once (ensures correct ordering)
    bands = aef_for_year(train_years[0], roi).bandNames().getInfo()
    frontier_bands = ["dist_to_nonforest_m", "dist_to_road_m"]
    train_selectors = bands + frontier_bands + ["label", "tYear"]
    unbiased_selectors = bands + frontier_bands + ["label", "tYear", "unbiased"]

    # 1) Balanced train exports (one task per year, or you can merge them)
    fc_all = None
    for y in train_years:
        fc_y = stratified_samples_for_year(
            t_year=y,
            region=roi,
            n_neg=args.n_neg,
            n_pos=args.n_pos,
            scale=args.scale,
            seed=args.seed,
            use_stable_label=args.use_stable_label,
        )
        fc_all = fc_y if fc_all is None else fc_all.merge(fc_y)

    desc = f"{args.prefix}_train_{train_years[0]}_{train_years[-1]}"
    fname = f"{args.prefix}_train_balanced_{train_years[0]}_{train_years[-1]}"
    task = export_fc_to_drive(fc_all, desc, fname, train_selectors, folder=args.drive_folder)
    print("Started:", desc, "| task id:", task.id)

    # 2) Unbiased forest-only export
    fc_u = unbiased_forest_samples(
        t_year=args.unbiased_year,
        region=roi,
        n_pixels=args.n_unbiased,
        scale=args.scale,
        seed=args.seed,
        use_stable_label=args.use_stable_label,
    )
    desc_u = f"{args.prefix}_unbiased_{args.unbiased_year}"
    fname_u = f"{args.prefix}_unbiased_forest_eval_{args.unbiased_year}"
    task_u = export_fc_to_drive(fc_u, desc_u, fname_u, unbiased_selectors, folder=args.drive_folder)
    print("Started:", desc_u, "| task id:", task_u.id)

    print("\nAll export tasks started. Check them in the Earth Engine Tasks tab (Code Editor) or in the EE Python task list.")


if __name__ == "__main__":
    main()
