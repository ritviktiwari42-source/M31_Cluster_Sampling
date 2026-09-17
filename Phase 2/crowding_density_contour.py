"""
crowding_density_contour.py

STANDALONE script -- a SEPARATE step from nearest_neighbor_crowding.py.

Takes the per-star table produced by nearest_neighbor_crowding.py
(nn_crowding_table.fits) and, for each magnitude bin:

  1. Builds a 2D density estimate (Gaussian KDE) of the point cloud
     from that bin's crowding plot -- separation to nearest star
     (log10 arcsec) vs. magnitude difference to that neighbor.
  2. Normalizes the density so its peak value = 1.
  3. Finds the 50% contour (i.e. the contour line where the
     normalized density equals 0.5) -- this traces the "envelope" of
     the point cloud that Dr. Ivanov described.
  4. Saves the (x, y) coordinates of that contour as a CSV table
     (separation in arcsec, magnitude difference), one row per point
     along the contour. If the contour has more than one disjoint
     loop, each is labeled with its own contour_id.
  5. Also saves a PNG per bin showing the normalized density map with
     the 50% contour drawn on top, as a visual check.

The KDE is computed in (log10(separation), magnitude difference)
space, since the crowding plots themselves use a log-scaled
separation axis -- this matches what you see by eye in those plots.
Contour coordinates are converted back to plain arcsec before being
saved to CSV, so the output is in physically meaningful units.

INPUT: nn_crowding_table.fits (from nearest_neighbor_crowding.py),
which must contain: nn_separation_arcsec, nn_mag_difference,
mag_bin_index.

USAGE:
    python crowding_density_contour.py
    (prompts for the input path -- no command-line arguments needed)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------
# 1. Build a normalized 2D density grid for one bin's point cloud,
#    using a SHARED x/y range across all bins (so every bin's plot
#    has identical axis limits and is directly comparable).
# ---------------------------------------------------------------
def compute_density_grid(log_sep, mag_diff, x_range, y_range, grid_size=200):
    """
    log_sep, mag_diff : 1D arrays for one magnitude bin
    x_range, y_range  : (lo, hi) tuples -- the SAME range used for
                         every bin, so all grids/plots line up.
    Returns (X_grid, Y_grid, density_norm) -- 2D meshgrids and the
    KDE density evaluated on them, normalized so the peak = 1.
    """
    kde = gaussian_kde(np.vstack([log_sep, mag_diff]))

    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    positions = np.vstack([X_grid.ravel(), Y_grid.ravel()])
    density = kde(positions).reshape(X_grid.shape)
    density_norm = density / density.max()

    return X_grid, Y_grid, density_norm


# ---------------------------------------------------------------
# 2. Extract the 50% contour's (x, y) points from the density grid.
# ---------------------------------------------------------------
def extract_contour_points(X_grid, Y_grid, density_norm, level=0.5):
    """
    Returns a list of (contour_id, log_sep_array, mag_diff_array)
    tuples -- one per disjoint closed/open loop found at the given
    normalized density level.
    """
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(X_grid, Y_grid, density_norm, levels=[level])
    plt.close(fig_tmp)

    loops = []
    # matplotlib >=3.8 uses cs.allsegs; older versions expose the same
    # attribute, so this works across versions without extra checks.
    segs = cs.allsegs[0]  # segments for our single requested level
    for i, seg in enumerate(segs):
        if len(seg) == 0:
            continue
        loops.append((i, seg[:, 0], seg[:, 1]))

    return loops


# ---------------------------------------------------------------
# 3. Plot: normalized density map + the 50% contour, per bin.
# ---------------------------------------------------------------
def plot_density_with_contour(X_grid, Y_grid, density_norm, loops,
                               mag_lo, mag_hi, n_stars, out_path,
                               x_range, y_range, filter_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(X_grid, Y_grid, density_norm, levels=20, cmap='viridis')
    fig.colorbar(cf, ax=ax, label='normalized density')

    for (cid, xs, ys) in loops:
        ax.plot(xs, ys, color='red', linewidth=2,
                label='50% contour' if cid == 0 else None)

    # Same axis limits on every bin's plot, so they're directly
    # comparable to each other.
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.invert_yaxis()

    # show x-axis in actual arcsec (it was computed in log10 space)
    tick_locs = ax.get_xticks()
    tick_locs = tick_locs[(tick_locs >= x_range[0]) & (tick_locs <= x_range[1])]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f'{10**t:.2g}' for t in tick_locs])

    ax.set_xlabel('separation to nearest star (arcsec)')
    ax.set_ylabel('magnitude difference to nearest star')
    ax.set_title(f'{filter_name}: mag {mag_lo:.1f}-{mag_hi:.1f} '
                 f'(N={n_stars}): '
                 f'density + 50% contour')
    if loops:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# Run the full density-contour pass for ONE filter, across all of
# its magnitude bins.
# ---------------------------------------------------------------
def process_one_filter(filter_name, nn_sep, mag, mag_diff, bin_index,
                        level, out_dir):
    filter_dir = os.path.join(out_dir, filter_name)
    os.makedirs(filter_dir, exist_ok=True)

    unique_bins = sorted(b for b in set(bin_index) if b >= 0)
    print(f"\n[{filter_name}] found {len(unique_bins)} magnitude bins.")

    # ---- First pass: figure out which bins are usable, and compute
    # ONE shared x/y range (with padding) across all of them for THIS
    # filter, so every bin's plot has identical axis limits. ----
    usable_bins = []
    all_log_sep = []
    all_mag_diff = []
    for b in unique_bins:
        in_bin = (bin_index == b)
        n_stars = int(in_bin.sum())
        if n_stars < 20:
            print(f"  [{filter_name}] bin index {b}: only {n_stars} stars "
                  f"-- skipping (too few for a reliable density estimate).")
            continue
        sep_bin = nn_sep[in_bin]
        mag_diff_bin = mag_diff[in_bin]
        valid = sep_bin > 0
        usable_bins.append(b)
        all_log_sep.append(np.log10(sep_bin[valid]))
        all_mag_diff.append(mag_diff_bin[valid])

    if not usable_bins:
        print(f"\n[{filter_name}] no bins had enough stars for a density "
              f"estimate.")
        return []

    pad_frac = 0.05
    concat_log_sep = np.concatenate(all_log_sep)
    concat_mag_diff = np.concatenate(all_mag_diff)
    x_lo, x_hi = np.min(concat_log_sep), np.max(concat_log_sep)
    y_lo, y_hi = np.min(concat_mag_diff), np.max(concat_mag_diff)
    x_pad = (x_hi - x_lo) * pad_frac
    y_pad = (y_hi - y_lo) * pad_frac
    x_range = (x_lo - x_pad, x_hi + x_pad)
    y_range = (max(0, y_lo - y_pad), y_hi + y_pad)
    print(f"[{filter_name}] shared axis range: separation "
          f"{10**x_range[0]:.3g}-{10**x_range[1]:.3g} arcsec, "
          f"mag diff {y_range[0]:.2f}-{y_range[1]:.2f}")

    contour_rows = []

    for b, log_sep, mag_diff_valid in zip(usable_bins, all_log_sep,
                                            all_mag_diff):
        in_bin = (bin_index == b)
        n_stars = int(in_bin.sum())
        mag_bin_vals = mag[in_bin]
        mag_lo, mag_hi = np.min(mag_bin_vals), np.max(mag_bin_vals)

        print(f"  [{filter_name}] bin {mag_lo:.1f}-{mag_hi:.1f} "
              f"({n_stars} stars): computing density...")

        X_grid, Y_grid, density_norm = compute_density_grid(
            log_sep, mag_diff_valid, x_range, y_range
        )
        loops = extract_contour_points(X_grid, Y_grid, density_norm,
                                        level=level)

        if not loops:
            print(f"    [warning] no contour found at level {level} for "
                  f"this bin -- the density may be too flat or too peaked.")

        png_path = os.path.join(
            filter_dir, f'density_contour_magbin_{mag_lo:.1f}_{mag_hi:.1f}.png'
        )
        plot_density_with_contour(X_grid, Y_grid, density_norm, loops,
                                   mag_lo, mag_hi, n_stars, png_path,
                                   x_range, y_range, filter_name)

        for (cid, log_sep_pts, mag_diff_pts) in loops:
            sep_pts = 10 ** log_sep_pts
            for sep_val, md_val in zip(sep_pts, mag_diff_pts):
                contour_rows.append({
                    'filter': filter_name,
                    'mag_bin_index': b,
                    'mag_bin_lo': mag_lo,
                    'mag_bin_hi': mag_hi,
                    'contour_id': cid,
                    'separation_arcsec': sep_val,
                    'mag_difference': md_val,
                })

        n_points = sum(len(xs) for (_, xs, _) in loops)
        print(f"    -> {len(loops)} contour loop(s), {n_points} points "
              f"total. Saved: {png_path}")

    return contour_rows


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def run_interactive():
    print("crowding_density_contour.py -- 50% density contour of the "
          "crowding cloud, per magnitude bin, for BOTH F475W and F814W\n")

    path = input(
        "Enter the path to nn_crowding_table.fits (from "
        "nearest_neighbor_crowding.py): "
    ).strip().strip('"').strip("'")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with fits.open(path) as hdul:
        data = hdul[1].data
        names = data.columns.names
        required = {'nn_separation_arcsec',
                    'mag_F475W', 'mag_F814W',
                    'nn_mag_difference_F475W', 'nn_mag_difference_F814W',
                    'mag_bin_index_F475W', 'mag_bin_index_F814W'}
        missing = required - set(names)
        if missing:
            print(f"Input file is missing required column(s): {missing}. "
                  f"Available columns: {names}. Did you generate this "
                  f"file with the current (dual-filter) version of "
                  f"nearest_neighbor_crowding.py?")
            sys.exit(1)

        nn_sep = np.array(data['nn_separation_arcsec'], dtype=float)
        mag_475 = np.array(data['mag_F475W'], dtype=float)
        mag_814 = np.array(data['mag_F814W'], dtype=float)
        mag_diff_475 = np.array(data['nn_mag_difference_F475W'], dtype=float)
        mag_diff_814 = np.array(data['nn_mag_difference_F814W'], dtype=float)
        bin_index_475 = np.array(data['mag_bin_index_F475W'], dtype=int)
        bin_index_814 = np.array(data['mag_bin_index_F814W'], dtype=int)

    level_input = input(
        "Contour density level to extract, as a fraction of the peak "
        "[default 0.5]: "
    ).strip()
    level = float(level_input) if level_input else 0.5

    out_dir = os.path.join(os.path.dirname(os.path.abspath(path)),
                            'density_contours')
    os.makedirs(out_dir, exist_ok=True)

    rows_475 = process_one_filter('F475W', nn_sep, mag_475, mag_diff_475,
                                    bin_index_475, level, out_dir)
    rows_814 = process_one_filter('F814W', nn_sep, mag_814, mag_diff_814,
                                    bin_index_814, level, out_dir)

    all_contour_rows = rows_475 + rows_814
    if all_contour_rows:
        contour_table = Table(rows=all_contour_rows)
        csv_path = os.path.join(out_dir, 'contour_50pct_points.csv')
        contour_table.write(csv_path, format='csv', overwrite=True)
        print(f"\nSaved all contour points (both filters) to: {csv_path} "
              f"({len(contour_table)} rows total)")
    else:
        print("\nNo contour points were found in any bin -- nothing to save.")

    print(f"\nAll density-map plots saved to: {out_dir}/F475W/ and "
          f"{out_dir}/F814W/")


if __name__ == "__main__":
    run_interactive()
