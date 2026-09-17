"""
select_background_regions.py

STANDALONE / REUSABLE script.

Input:  a FITS table produced by your DBSCAN export step, containing
        at minimum these columns:
            X              -- arcmin offset position
            Y              -- arcmin offset position
            cluster_label  -- DBSCAN label (-1 = noise, 0,1,2,... = cluster)

Output: for every real cluster (label >= 0) in the file, this script:
    1. Computes the cluster's own center (mean X, Y) and radius
       (distance to its furthest member star) -- same definition as
       the paper's "each field with the same area as the cluster".
    2. Places up to 10 background regions in a ring around that
       cluster, each with the SAME radius as the cluster itself.
    3. Automatically skips/repositions any candidate region that
       would overlap the cluster itself OR any other cluster.
    4. Reports each region's star count, so you can sanity-check
       whether a region is too sparse to be a useful comparison field.
    5. Saves the region list as BOTH a CSV file and a FITS table.
    6. Saves a PNG plot showing every star, every cluster (circled),
       and every background region (circled), so you can visually
       verify the geometry before running any CMD matching on it.
    7. Returns everything as a dictionary in memory if you're calling
       this from another script.

This script does NOT run DBSCAN and does NOT depend on any of your
earlier scripts -- it only needs the exported FITS table's four
columns (X, Y, cluster_label; RA/Dec are not required for this step).

USAGE (command line):
    python select_background_regions.py /path/to/dbscan_export.fits

USAGE (as a module):
    from select_background_regions import select_all_background_regions
    results = select_all_background_regions('dbscan_export.fits')
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # safe for headless/no-display environments
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits
from astropy.table import Table


# ---------------------------------------------------------------
# 1. Determine each cluster's center and radius directly from the
#    DBSCAN labels in the file.
# ---------------------------------------------------------------
def get_cluster_geometry(X, Y, cluster_label):
    """
    Returns a dict: {cluster_id: {'center': (cx, cy), 'radius': r,
                                   'n_stars': N}}
    for every cluster_label >= 0 present in the data.
    """
    geometry = {}
    unique_labels = sorted(set(cluster_label))

    for k in unique_labels:
        if k == -1:
            continue  # skip noise
        mask = (cluster_label == k)
        cx = np.mean(X[mask])
        cy = np.mean(Y[mask])
        radius = np.max(np.sqrt((X[mask] - cx) ** 2 + (Y[mask] - cy) ** 2))
        geometry[k] = {
            'center': (cx, cy),
            'radius': radius,
            'n_stars': int(mask.sum()),
        }
    return geometry


# ---------------------------------------------------------------
# 2. Overlap checks: a candidate region must not overlap (a) any
#    cluster's own circle, and (b) any background region already
#    placed for this cluster -- so all N regions stay independent,
#    non-overlapping samples of the field.
# ---------------------------------------------------------------
def region_overlaps_any_cluster(fx, fy, radius, geometry, exclude_label=None):
    """
    Returns True if a circle of the given radius, centered at (fx, fy),
    would overlap any cluster's own circle (distance between centers
    < sum of the two radii means overlap).
    """
    for k, info in geometry.items():
        if k == exclude_label:
            # still check against the cluster we're placing regions for,
            # just using its own known center/radius -- so don't
            # actually skip it; this branch is unused but kept for clarity
            pass
        cx, cy = info['center']
        cluster_r = info['radius']
        dist = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
        if dist < (radius + cluster_r):
            return True
    return False


def region_overlaps_existing_regions(fx, fy, radius, existing_regions):
    """
    Returns True if a circle of the given radius, centered at (fx, fy),
    would overlap any already-placed background region in
    `existing_regions` (a list of dicts with 'x', 'y', 'radius').
    """
    for reg in existing_regions:
        dist = np.sqrt((fx - reg['x']) ** 2 + (fy - reg['y']) ** 2)
        if dist < (radius + reg['radius']):
            return True
    return False


# ---------------------------------------------------------------
# 3. Place up to N background regions around ONE cluster, using that
#    cluster's own radius, skipping placements that overlap any
#    cluster OR any background region already placed for this cluster
#    -- guaranteeing all N regions are mutually non-overlapping.
# ---------------------------------------------------------------
def place_background_regions(cluster_id, geometry, X, Y,
                               n_regions=10, offset_factor=3.0,
                               max_attempts=500, seed=None):
    """
    Returns a list of dicts, one per successfully placed region:
        {'x': fx, 'y': fy, 'radius': r, 'n_stars': count}
    """
    rng = np.random.default_rng(seed if seed is not None else cluster_id)
    cx, cy = geometry[cluster_id]['center']
    r = geometry[cluster_id]['radius']

    placed_regions = []
    attempts = 0

    while len(placed_regions) < n_regions and attempts < max_attempts:
        theta = rng.uniform(0, 2 * np.pi)
        # Randomize the offset distance a bit too, so rejected angles
        # get a genuinely different retry rather than the same spot.
        # As more regions get placed, gradually allow slightly larger
        # offsets so later regions have room to avoid earlier ones.
        growth = 1.0 + 0.15 * len(placed_regions)
        this_offset = offset_factor * growth * (1.0 + 0.3 * rng.uniform(-1, 1))
        fx = cx + this_offset * r * np.cos(theta)
        fy = cy + this_offset * r * np.sin(theta)

        attempts += 1

        if region_overlaps_any_cluster(fx, fy, r, geometry):
            continue
        if region_overlaps_existing_regions(fx, fy, r, placed_regions):
            continue

        dist = np.sqrt((X - fx) ** 2 + (Y - fy) ** 2)
        n_stars = int((dist <= r).sum())

        placed_regions.append({
            'x': fx, 'y': fy, 'radius': r, 'n_stars': n_stars
        })

    if len(placed_regions) < n_regions:
        print(f"  [warning] cluster {cluster_id}: only placed "
              f"{len(placed_regions)}/{n_regions} non-overlapping "
              f"regions after {attempts} attempts. Consider increasing "
              f"offset_factor or max_attempts.")

    return placed_regions


# ---------------------------------------------------------------
# 4. Plot: every star, colored by role -- noise (gray), each cluster
#    in its OWN distinct color (so cluster 0 looks different from
#    cluster 1, etc., same as the original DBSCAN plot), and every
#    background-region star in a single shared green. No boundary
#    circles; the colored points themselves show the selection.
# ---------------------------------------------------------------
def plot_regions(X, Y, cluster_label, geometry, all_regions, out_path,
                  bg_cluster_id=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    if bg_cluster_id is None:
        # Fallback: recompute membership if not passed in directly.
        n_stars = len(X)
        bg_cluster_id = np.full(n_stars, -1, dtype=int)
        for cluster_id, regions in all_regions.items():
            for reg in regions:
                dist = np.sqrt((X - reg['x']) ** 2 + (Y - reg['y']) ** 2)
                in_region = (dist <= reg['radius']) & (bg_cluster_id == -1)
                bg_cluster_id[in_region] = cluster_id

    noise_mask = (cluster_label == -1) & (bg_cluster_id == -1)
    bg_mask = (bg_cluster_id != -1) & (cluster_label == -1)

    # plain field/noise stars, drawn first (bottom layer)
    ax.plot(X[noise_mask], Y[noise_mask], '.', markersize=1,
            color='lightgray', zorder=1, label='noise')

    # background-region stars, all one shared green, drawn above noise
    ax.plot(X[bg_mask], Y[bg_mask], '.', markersize=3,
            color='limegreen', zorder=2, label='background region stars')

    # each cluster gets its own distinct color (Spectral colormap,
    # same style as the original DBSCAN plot), drawn on top
    cluster_ids = sorted(geometry.keys())
    colors = plt.cm.Spectral(np.linspace(0, 1, len(cluster_ids)))
    for cluster_id, col in zip(cluster_ids, colors):
        mask = (cluster_label == cluster_id)
        ax.plot(X[mask], Y[mask], 'o', markersize=3,
                markerfacecolor=col, markeredgecolor='none',
                zorder=3, label=f'cluster {cluster_id}')

        cx, cy = geometry[cluster_id]['center']
        r = geometry[cluster_id]['radius']
        ax.text(cx, cy + r * 1.6, f"C{cluster_id}", color='black',
                fontsize=8, ha='center', zorder=5)

    ax.set_xlabel('X, arcmin')
    ax.set_ylabel('Y, arcmin')
    ax.set_title('DBSCAN clusters (individually colored) and '
                 'background-region stars (green)')
    ax.invert_xaxis()  # match RA-style flip used elsewhere in this project
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper right', markerscale=4, fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# 4b. Build a FULL per-star table (same star count as the input file)
#     with two new columns marking background-region membership, so
#     loading this in TOPCAT and plotting X vs Y reproduces the same
#     picture as the PNG (clusters + background regions), not just a
#     10-row-per-cluster region-definition table.
# ---------------------------------------------------------------
def build_full_star_table(full_star_table, X, Y, all_regions):
    """
    full_star_table : astropy Table with all original columns
                       (ra, dec, X, Y, color, mag, cluster_label, ...)
    X, Y            : arrays matching full_star_table's rows
    all_regions     : dict {cluster_id: [ {x, y, radius, n_stars}, ... ]}

    Adds two new columns to a COPY of full_star_table:
        bg_cluster_id    : which cluster's background-region set this
                            star belongs to (-1 if none)
        bg_region_index  : which region (1..N) within that cluster's
                            set (-1 if none)

    A star can only belong to one background region under this
    script's placement logic (regions are non-overlapping within a
    cluster's own set); if regions from two DIFFERENT clusters happen
    to overlap (not checked against each other), the first match found
    wins and is reported.

    Returns the augmented Table.
    """
    out_table = full_star_table.copy()
    n_stars = len(out_table)

    bg_cluster_id = np.full(n_stars, -1, dtype=int)
    bg_region_index = np.full(n_stars, -1, dtype=int)

    for cluster_id, regions in all_regions.items():
        for i, reg in enumerate(regions):
            dist = np.sqrt((X - reg['x']) ** 2 + (Y - reg['y']) ** 2)
            in_region = (dist <= reg['radius']) & (bg_cluster_id == -1)
            bg_cluster_id[in_region] = cluster_id
            bg_region_index[in_region] = i + 1

    out_table['bg_cluster_id'] = bg_cluster_id
    out_table['bg_region_index'] = bg_region_index

    return out_table


# ---------------------------------------------------------------
# 5. Do this for every cluster in the file, and report/save results.
# ---------------------------------------------------------------
def select_all_background_regions(fits_path, n_regions=10, offset_factor=3.0,
                                    save_csv=True, save_fits=True,
                                    save_png=True, save_full_fits=True,
                                    seed=None):
    """
    fits_path : path to your DBSCAN-exported FITS table (must contain
                X, Y, cluster_label columns)
    n_regions : how many background regions to place per cluster (<=10
                per your request; fewer will be placed if the geometry
                doesn't allow non-overlapping placement)
    offset_factor : how many cluster-radii away (on average) to place
                region centers. Increase this if regions keep
                overlapping neighboring clusters.
    save_csv  : if True, writes 'background_regions.csv' next to the
                input file, one row per region.
    save_fits : if True, writes 'background_regions.fits' next to the
                input file, same content as the CSV.
    save_png  : if True, writes 'background_regions.png' next to the
                input file -- a plot of all stars, cluster circles
                (red), and background-region circles (green).
    save_full_fits : if True, writes 'background_regions_full.fits' --
                a FULL per-star table (same rows as the input file)
                with two added columns, 'bg_cluster_id' and
                'bg_region_index', marking which background region (if
                any) each star belongs to. Load THIS file into TOPCAT
                (not background_regions.fits) to reproduce the same
                picture as the PNG.

    Returns
    -------
    all_regions : dict {cluster_id: [ {x, y, radius, n_stars}, ... ]}
    geometry    : dict {cluster_id: {center, radius, n_stars}}
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data

        required_cols = {'X', 'Y', 'cluster_label'}
        missing = required_cols - set(data.columns.names)
        if missing:
            raise ValueError(
                f"Input FITS file is missing required column(s): {missing}. "
                f"Available columns: {data.columns.names}"
            )

        # Copy out of the memory-mapped FITS record array so the arrays
        # remain valid/writable after the file handle above is closed.
        X = np.array(data['X'], dtype=float)
        Y = np.array(data['Y'], dtype=float)
        cluster_label = np.array(data['cluster_label'], dtype=int)

        # Keep a full copy of every original column too (ra, dec, color,
        # mag, etc.) so the "full" output file below can include them
        # alongside the new background-region columns.
        full_star_table = Table(data)

    geometry = get_cluster_geometry(X, Y, cluster_label)
    print(f"Found {len(geometry)} clusters in {os.path.basename(fits_path)}\n")

    all_regions = {}
    csv_rows = []

    for cluster_id, info in geometry.items():
        cx, cy = info['center']
        r = info['radius']
        n_cluster_stars = info['n_stars']

        print(f"Cluster {cluster_id}: center=({cx:.4f}, {cy:.4f}), "
              f"radius={r:.4f}, n_stars={n_cluster_stars}")

        regions = place_background_regions(
            cluster_id, geometry, X, Y,
            n_regions=n_regions, offset_factor=offset_factor, seed=seed
        )
        all_regions[cluster_id] = regions

        for i, reg in enumerate(regions):
            print(f"    bg{i+1}: center=({reg['x']:.4f}, {reg['y']:.4f}), "
                  f"radius={reg['radius']:.4f}, n_stars={reg['n_stars']}")
            csv_rows.append({
                'cluster_id': cluster_id,
                'region_index': i + 1,
                'x_center': reg['x'],
                'y_center': reg['y'],
                'radius': reg['radius'],
                'n_stars': reg['n_stars'],
            })
        print()

    out_dir = os.path.dirname(os.path.abspath(fits_path))
    region_table = Table(rows=csv_rows) if csv_rows else Table(
        names=['cluster_id', 'region_index', 'x_center', 'y_center',
               'radius', 'n_stars'],
        dtype=[int, int, float, float, float, int]
    )

    if save_csv:
        csv_path = os.path.join(out_dir, 'background_regions.csv')
        region_table.write(csv_path, format='csv', overwrite=True)
        print(f"Saved region list (CSV) to: {csv_path}")

    if save_fits:
        fits_out_path = os.path.join(out_dir, 'background_regions.fits')
        if os.path.exists(fits_out_path):
            os.remove(fits_out_path)
        hdu = fits.BinTableHDU(data=region_table.as_array(),
                                name='BACKGROUND_REGIONS')
        hdu.writeto(fits_out_path)
        print(f"Saved region list (FITS) to: {fits_out_path}")

    if save_png:
        png_path = os.path.join(out_dir, 'background_regions.png')
        full_table_for_plot = build_full_star_table(full_star_table, X, Y,
                                                       all_regions)
        plot_regions(X, Y, cluster_label, geometry, all_regions, png_path,
                     bg_cluster_id=np.array(full_table_for_plot['bg_cluster_id']))
        print(f"Saved region plot (PNG) to: {png_path}")

    if save_full_fits:
        full_table = build_full_star_table(full_star_table, X, Y, all_regions)
        full_fits_path = os.path.join(out_dir, 'background_regions_full.fits')
        if os.path.exists(full_fits_path):
            os.remove(full_fits_path)
        hdu = fits.BinTableHDU(data=full_table.as_array(),
                                name='STARS_WITH_BG_REGIONS')
        hdu.writeto(full_fits_path)
        print(f"Saved full per-star table (FITS) to: {full_fits_path}")
        print("  -> load THIS file in TOPCAT (not background_regions.fits) "
              "to reproduce the PNG picture: plot X vs Y, then colour by "
              "'cluster_label' for clusters and/or 'bg_cluster_id' for "
              "background-region membership.")

    return all_regions, geometry


# ---------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Path given as a command-line argument, e.g.:
        #   python select_background_regions.py /path/to/file.fits
        fits_path = sys.argv[1]
        n_regions = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        offset_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    else:
        # No arguments given (e.g. running via an IDE's "Run" button) --
        # ask for the path interactively instead.
        fits_path = input(
            "Enter the path to your DBSCAN-exported FITS file "
            "(e.g. ~/Documents/ESO_Work/dbscan_export_for_topcat.fits): "
        ).strip().strip('"').strip("'")
        fits_path = os.path.expanduser(fits_path)

        n_regions_input = input(
            "Number of background regions per cluster [default 10]: "
        ).strip()
        n_regions = int(n_regions_input) if n_regions_input else 10

        offset_input = input(
            "Offset factor -- how far to place regions, in units of "
            "cluster radius [default 3.0]: "
        ).strip()
        offset_factor = float(offset_input) if offset_input else 3.0

    if not os.path.isfile(fits_path):
        print(f"File not found: {fits_path}")
        sys.exit(1)

    select_all_background_regions(
        fits_path, n_regions=n_regions, offset_factor=offset_factor
    )
