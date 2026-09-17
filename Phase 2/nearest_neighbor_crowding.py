"""
nearest_neighbor_crowding.py

Investigates how close two stars can be to each other before they'd
blend/merge, as a function of their magnitudes -- the analysis Dr.
Ivanov described. This version considers BOTH filters used in this
project, F475W and F814W, since the blending/crowding limit can differ
between them (a pair of stars might be hard to separate in one filter
but easier to distinguish in the other, depending on their spectral
shape).

  1. Find each star's nearest neighbor ONCE, using a KDTree over
     sky position only -- "who is closest to whom" is a purely
     spatial question and does not depend on which filter you look
     at, so the same neighbor pairing is reused for both filters.
  2. For EACH filter (F475W and F814W) separately:
       a. Sort stars by that filter's magnitude, brightest to
          faintest, and split into magnitude bins (default 2 mag
          wide).
       b. Compute the magnitude difference to the nearest neighbor,
          IN THAT FILTER.
       c. For each bin, plot separation (arcsec) vs. that filter's
          magnitude difference.
  3. Save one combined per-star table with both filters' magnitudes,
     magnitude differences, and bin indices -- what the next step
     (parametrizing the envelope) will need, for either filter.

This is a standalone, general-purpose tool -- it does not depend on
DBSCAN, clusters, or any other part of the pipeline. It just needs a
catalog of star positions and magnitudes in both filters.

INPUT: a FITS file with position columns, and magnitudes in BOTH
filters, detected in this order of preference:
  1. 'f475w_vega' and 'f814w_vega' columns directly (e.g.
     phast_subset.fits / phast_subset_cleaned.fits), OR
  2. 'color' and 'mag' columns (e.g. dbscan_export_for_topcat.fits /
     background_regions_full.fits from this project) -- in which case
     F814W is reconstructed as mag, and F475W is reconstructed as
     color + mag (since color = F475W - F814W throughout this
     project).
Position: either 'X'/'Y' (arcmin offsets) or 'ra'/'dec' (degrees).

USAGE:
    python nearest_neighbor_crowding.py
    (prompts for everything -- no command-line arguments needed)

TIP (per Dr. Ivanov's suggestion): start with a smaller chunk of your
catalog to get the code and plots right quickly, then re-run on the
full phast_subset.fits catalog once you're happy with it. The script
lets you optionally subsample for a quick test run.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

FILTER_NAMES = ['F475W', 'F814W']


# ---------------------------------------------------------------
# 1. Load the catalog -- positions (converted to arcsec offsets) and
#    BOTH filters' magnitudes.
# ---------------------------------------------------------------
def load_catalog(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        names = data.columns.names

        if 'f475w_vega' in names and 'f814w_vega' in names:
            mag_475 = np.array(data['f475w_vega'], dtype=float)
            mag_814 = np.array(data['f814w_vega'], dtype=float)
            source_note = "read directly from 'f475w_vega' / 'f814w_vega'"
        elif 'color' in names and 'mag' in names:
            mag_814 = np.array(data['mag'], dtype=float)
            color = np.array(data['color'], dtype=float)
            mag_475 = color + mag_814
            source_note = ("reconstructed from 'color' + 'mag' "
                            "(F814W = mag, F475W = color + mag)")
        else:
            raise ValueError(
                f"Need either ('f475w_vega','f814w_vega') or "
                f"('color','mag') columns to get both filters. "
                f"Available columns: {names}"
            )

        if 'X' in names and 'Y' in names:
            X_arcsec = np.array(data['X'], dtype=float) * 60.0
            Y_arcsec = np.array(data['Y'], dtype=float) * 60.0
        elif 'ra' in names and 'dec' in names:
            ra = np.array(data['ra'], dtype=float)
            dec = np.array(data['dec'], dtype=float)
            dec0 = np.mean(dec)
            cos_dec0 = np.cos(np.deg2rad(dec0))
            ra0 = np.mean(ra)
            X_arcsec = (ra - ra0) * cos_dec0 * 3600.0
            Y_arcsec = (dec - dec0) * 3600.0
        else:
            raise ValueError(
                f"Need either ('X','Y') or ('ra','dec') columns. "
                f"Available columns: {names}"
            )

    return X_arcsec, Y_arcsec, mag_475, mag_814, source_note


# ---------------------------------------------------------------
# 2. Nearest-neighbor search via KDTree -- PURELY SPATIAL, computed
#    once and reused for both filters (who your nearest neighbor is
#    doesn't depend on which filter you're measuring brightness in).
# ---------------------------------------------------------------
def compute_nearest_neighbor_positions(X_arcsec, Y_arcsec):
    positions = np.column_stack([X_arcsec, Y_arcsec])
    tree = cKDTree(positions)
    dist, idx = tree.query(positions, k=2)
    nn_dist = dist[:, 1]
    nn_idx = idx[:, 1]
    return nn_dist, nn_idx


def compute_mag_diff(mag, nn_idx):
    """Magnitude difference to the (already-found) nearest neighbor,
    in whichever filter `mag` is given in."""
    return np.abs(mag[nn_idx] - mag)


# ---------------------------------------------------------------
# 3. Magnitude bins
# ---------------------------------------------------------------
def make_magnitude_bins(mag, bin_width=2.0, mag_min=None, mag_max=None):
    lo = mag_min if mag_min is not None else np.min(mag)
    hi = mag_max if mag_max is not None else np.max(mag)
    lo = np.floor(lo / bin_width) * bin_width
    hi = np.ceil(hi / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    return edges


# ---------------------------------------------------------------
# 4. Plot one magnitude bin, for one filter: separation (arcsec, log
#    scale) vs. magnitude difference to the nearest neighbor.
# ---------------------------------------------------------------
def plot_bin(nn_dist_bin, mag_diff_bin, mag_lo, mag_hi, filter_name,
             out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(nn_dist_bin, mag_diff_bin, s=4, alpha=0.35, color='red',
               edgecolor='none')
    ax.set_xscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('separation to nearest star (arcsec)')
    ax.set_ylabel(f'{filter_name} magnitude difference to nearest star')
    ax.set_title(f'{filter_name}: mag {mag_lo:.1f}-{mag_hi:.1f}  '
                 f'(N={len(nn_dist_bin)})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# 5. Run the full binning + plotting pass for ONE filter.
# ---------------------------------------------------------------
def process_one_filter(filter_name, mag, nn_dist, nn_idx, bin_width,
                        mag_min, mag_max, out_dir):
    mag_diff = compute_mag_diff(mag, nn_idx)
    edges = make_magnitude_bins(mag, bin_width=bin_width,
                                 mag_min=mag_min, mag_max=mag_max)
    print(f"\n[{filter_name}] magnitude bins ({bin_width} mag wide): "
          f"{edges[0]:.1f} to {edges[-1]:.1f}")

    filter_dir = os.path.join(out_dir, filter_name)
    os.makedirs(filter_dir, exist_ok=True)

    bin_label = np.full(len(mag), -1, dtype=int)

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (mag >= lo) & (mag < hi)
        n_in_bin = in_bin.sum()
        bin_label[in_bin] = i

        if n_in_bin == 0:
            continue

        print(f"  [{filter_name}] bin {lo:.1f}-{hi:.1f}: {n_in_bin} stars")
        out_path = os.path.join(
            filter_dir, f'nn_separation_magbin_{lo:.1f}_{hi:.1f}.png'
        )
        plot_bin(nn_dist[in_bin], mag_diff[in_bin], lo, hi, filter_name,
                 out_path)

    return mag_diff, bin_label


# ---------------------------------------------------------------
# Main interactive driver
# ---------------------------------------------------------------
def run_interactive():
    print("nearest_neighbor_crowding.py -- separation vs. magnitude "
          "difference, binned by brightness, for BOTH F475W and F814W\n")

    path = input(
        "Enter the path to your catalog FITS file (e.g. "
        "dbscan_export_for_topcat.fits for a quick test chunk, or "
        "phast_subset.fits for the full catalog): "
    ).strip().strip('"').strip("'")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    X_arcsec, Y_arcsec, mag_475, mag_814, source_note = load_catalog(path)
    n_total = len(mag_475)
    print(f"\nLoaded {n_total} stars. F475W and F814W magnitudes "
          f"{source_note}.")
    print(f"F475W range: {np.min(mag_475):.2f} to {np.max(mag_475):.2f}")
    print(f"F814W range: {np.min(mag_814):.2f} to {np.max(mag_814):.2f}")

    sub_input = input(
        "\nSubsample size for a quick test run (e.g. 5000), or press "
        "Enter to use all stars: "
    ).strip()
    if sub_input:
        n_sub = int(sub_input)
        if n_sub < n_total:
            rng = np.random.default_rng(42)
            sel = rng.choice(n_total, size=n_sub, replace=False)
            X_arcsec, Y_arcsec = X_arcsec[sel], Y_arcsec[sel]
            mag_475, mag_814 = mag_475[sel], mag_814[sel]
            print(f"Using a random subsample of {n_sub} stars.")

    def ask_range(filter_name):
        range_input = input(
            f"Restrict {filter_name} to a magnitude range, e.g. '18,30' "
            f"(press Enter to use the full range found -- note: extreme "
            f"outlier values, like bad-photometry placeholders, will "
            f"distort the bins if included): "
        ).strip()
        if not range_input:
            return None, None
        try:
            lo, hi = [float(v) for v in range_input.split(',')]
            return lo, hi
        except ValueError:
            print("  Could not parse that range -- using the full range.")
            return None, None

    mag_min_475, mag_max_475 = ask_range('F475W')
    mag_min_814, mag_max_814 = ask_range('F814W')

    # A star must be reasonable in BOTH filters to be kept, otherwise
    # its (reconstructed) color/other-filter value would be nonsense.
    keep = np.ones(len(mag_475), dtype=bool)
    if mag_min_475 is not None:
        keep &= (mag_475 >= mag_min_475) & (mag_475 <= mag_max_475)
    if mag_min_814 is not None:
        keep &= (mag_814 >= mag_min_814) & (mag_814 <= mag_max_814)
    if not keep.all():
        X_arcsec, Y_arcsec = X_arcsec[keep], Y_arcsec[keep]
        mag_475, mag_814 = mag_475[keep], mag_814[keep]
        print(f"\nAfter magnitude-range restriction(s): {len(mag_475)} "
              f"stars remain.")

    bin_width_input = input(
        "Magnitude bin width (used for both filters) [default 2.0]: "
    ).strip()
    bin_width = float(bin_width_input) if bin_width_input else 2.0

    print(f"\nBuilding KDTree and finding nearest neighbors (spatial, "
          f"filter-independent) for {len(mag_475)} stars...")
    nn_dist, nn_idx = compute_nearest_neighbor_positions(X_arcsec, Y_arcsec)
    print("Done.")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(path)),
                            'nn_crowding')
    os.makedirs(out_dir, exist_ok=True)

    mag_diff_475, bin_label_475 = process_one_filter(
        'F475W', mag_475, nn_dist, nn_idx, bin_width,
        mag_min_475, mag_max_475, out_dir
    )
    mag_diff_814, bin_label_814 = process_one_filter(
        'F814W', mag_814, nn_dist, nn_idx, bin_width,
        mag_min_814, mag_max_814, out_dir
    )

    # Save ONE combined per-star table covering both filters -- what
    # the next step (parametrizing the envelope) will need, for
    # either filter.
    out_table = Table({
        'X_arcsec': X_arcsec,
        'Y_arcsec': Y_arcsec,
        'mag_F475W': mag_475,
        'mag_F814W': mag_814,
        'nn_separation_arcsec': nn_dist,
        'nn_mag_difference_F475W': mag_diff_475,
        'nn_mag_difference_F814W': mag_diff_814,
        'mag_bin_index_F475W': bin_label_475,
        'mag_bin_index_F814W': bin_label_814,
    })
    table_path = os.path.join(out_dir, 'nn_crowding_table.fits')
    if os.path.exists(table_path):
        os.remove(table_path)
    hdu = fits.BinTableHDU(data=out_table.as_array(), name='NN_CROWDING')
    hdu.writeto(table_path)

    print(f"\nSaved bin plots for both filters (in {out_dir}/F475W/ and "
          f"{out_dir}/F814W/) and the combined per-star table.")
    print(f"Master table: {table_path}")


if __name__ == "__main__":
    run_interactive()
