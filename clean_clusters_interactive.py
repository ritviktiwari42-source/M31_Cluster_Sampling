"""
clean_clusters_interactive.py

STANDALONE / REUSABLE script -- the final "leftover" step after:
    1. M31_clusterfinder_101_updated.py  (DBSCAN clustering)
    2. export_for_topcat.py              (-> dbscan_export_for_topcat.fits)
    3. select_background_regions.py      (-> background_regions_full.fits)

Unlike clean_clusters.py (which processes every cluster automatically in
one pass), THIS script walks through your clusters ONE AT A TIME,
interactively:

    For each cluster:
      1. Build its raw CMD (color, mag).
      2. Ask your permission before normalizing, and ask which of the
         three distance formulas to use:
           'simple'      Dist = sqrt( (dColor)^2 + (dMag)^2 )
           'minmax'      normalized by (max - min) of the CLUSTER's
                         own color/mag range
           'percentile'  normalized by the cluster's 5th-95th
                         percentile color/mag range
      3. Save the normalized CMD plot for the cluster (used for every
         step below -- the SAME normalization is applied consistently
         to the cluster and to every background region).
      4. Save one PNG per background region, each showing the cluster's
         normalized CMD overlaid with that region's normalized CMD.
      5. Run the nearest-neighbor field-star matching across all of the
         cluster's regions, compute membership probability per star,
         and save the final cleaned CMD (members only).
      6. Ask whether to continue to the next cluster, jump to a specific
         cluster ID, or stop.

INPUT: background_regions_full.fits (from select_background_regions.py),
which must contain: ra, dec, X, Y, color, mag, cluster_label,
bg_cluster_id, bg_region_index.

OUTPUTS (all saved next to the input file, organized in one subfolder
per cluster -- e.g. cluster_07/):
    raw_cmd_cluster_<id>.png              -- non-normalized CMD, unfiltered
    normalized_cmd_cluster_<id>.png       -- output (1)
    region_<ri>_cmd_cluster_<id>.png      -- output (2), one per region
    luminosity_function_cluster_<id>.png  -- LF: cluster area (all stars)
                                              vs. background area (avg per
                                              region), for direct comparison
    cleaned_cmd_cluster_<id>.png          -- output (3), normalized
    raw_cleaned_cmd_cluster_<id>.png      -- non-normalized CMD, cleaned
    raw_removed_cmd_cluster_<id>.png      -- non-normalized CMD of the
                                              REMOVED field stars
    three_panel_cmd_cluster_<id>.png      -- all stars / cluster / field,
                                              side by side (paper Fig. 4
                                              style)
    member_field_cmd_cluster_<id>.png     -- single CMD, members (red)
                                              + field stars (blue)
    probability_cmd_cluster_<id>.png      -- single CMD, all stars
                                              colored by continuous
                                              membership probability
    membership_cluster_<id>.csv           -- per-star coordinates (RA/Dec
                                              in degrees, X/Y in arcmin),
                                              photometry, probabilities,
                                              and membership flag

USAGE:
    python clean_clusters_interactive.py
    (it will prompt you for the FITS file path and walk you through
    each cluster interactively -- no command-line arguments needed)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_star_subtraction import (
    get_normalization_scales, normalize_cmd, compute_membership_probabilities
)


# Filter names actually used to build 'color' and 'mag' in this pipeline
# (color = f475w_vega - f814w_vega, mag = f814w_vega). Shown on every
# plot's axis labels instead of the generic words 'color'/'mag'. If you
# change which filters feed the color/mag columns upstream, update these
# two constants to match.
COLOR_LABEL = 'F475W - F814W (color)'
MAG_LABEL = 'F814W (mag)'
MAG_FILTER_NAME = 'F814W'


VALID_METHODS = {'1': 'simple', '2': 'minmax', '3': 'percentile',
                  'simple': 'simple', 'minmax': 'minmax',
                  'percentile': 'percentile'}


# ---------------------------------------------------------------
# Small interactive helpers
# ---------------------------------------------------------------
def ask_yes_no(prompt, default_yes=True):
    suffix = " [Y/n]: " if default_yes else " [y/N]: "
    ans = input(prompt + suffix).strip().lower()
    if ans == '':
        return default_yes
    return ans.startswith('y')


def ask_normalization_method():
    print("  Choose a normalization method:")
    print("    [1] simple     -- Dist = sqrt(dColor^2 + dMag^2), no scaling")
    print("    [2] minmax     -- scale by (max - min) of the cluster's own range")
    print("    [3] percentile -- scale by the cluster's 5th-95th percentile range")
    while True:
        ans = input("  Enter 1, 2, or 3 (or the method name): ").strip().lower()
        if ans in VALID_METHODS:
            return VALID_METHODS[ans]
        print("  Not recognized -- please enter 1, 2, 3, 'simple', "
              "'minmax', or 'percentile'.")


# ---------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------
def plot_normalized_cmd(color_n, mag_n, cluster_id, method, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(color_n, mag_n, s=12, color='crimson')
    ax.invert_yaxis()
    ax.set_xlabel(COLOR_LABEL)
    ax.set_ylabel(MAG_LABEL)
    ax.set_title(f'Cluster {cluster_id}: normalized CMD ({method})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cluster_plus_region(color_n, mag_n, region_color_n, region_mag_n,
                              cluster_id, region_index, method, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(color_n, mag_n, s=14, color='crimson',
               label=f'cluster {cluster_id}', zorder=2)
    ax.scatter(region_color_n, region_mag_n, s=14, color='limegreen',
               label=f'background region {region_index}', zorder=1)
    ax.invert_yaxis()
    ax.set_xlabel(COLOR_LABEL)
    ax.set_ylabel(MAG_LABEL)
    ax.set_title(f'Cluster {cluster_id} + background region {region_index}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cleaned_cmd(color_n, mag_n, member_mask, cluster_id, method,
                      out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(color_n[member_mask], mag_n[member_mask], s=14,
               color='crimson')
    ax.invert_yaxis()
    ax.set_xlabel(COLOR_LABEL)
    ax.set_ylabel(MAG_LABEL)
    ax.set_title(f'Cluster {cluster_id}: cleaned CMD '
                 f'({member_mask.sum()}/{len(member_mask)} members kept)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_raw_cmd(color_raw, mag_raw, cluster_id, out_path, title_suffix=''):
    """Plain (non-normalized) CMD -- raw color and mag, as measured,
    no rescaling of either axis."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(color_raw, mag_raw, s=14, color='crimson')
    ax.invert_yaxis()
    ax.set_xlabel(COLOR_LABEL)
    ax.set_ylabel(MAG_LABEL)
    ax.set_title(f'Cluster {cluster_id}: raw CMD{title_suffix}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_three_panel_cmd(cluster_color, cluster_mag, member_mask,
                          cluster_id, out_path):
    """
    Three-panel, non-normalized CMD comparison, in the style of
    Ivanov et al. (2005) Fig. 4 (left/middle/right panels, shared axes,
    star counts given in brackets):
      1. All stars in the cluster area -- uncleaned, unfiltered.
      2. Cluster only -- the cleaned members that survived the
         membership-probability cut.
      3. Field only -- the stars removed as likely contamination.
    All three panels share the same x/y limits so they are directly
    comparable by eye, exactly as in the paper.
    """
    removed_mask = ~member_mask

    panels = [
        (cluster_color, cluster_mag, 'All stars'),
        (cluster_color[member_mask], cluster_mag[member_mask], 'Cluster'),
        (cluster_color[removed_mask], cluster_mag[removed_mask], 'Field'),
    ]

    # shared axis limits across all three panels, with a small margin
    x_all, y_all = cluster_color, cluster_mag
    x_margin = 0.05 * (np.max(x_all) - np.min(x_all) + 1e-9)
    y_margin = 0.05 * (np.max(y_all) - np.min(y_all) + 1e-9)
    xlim = (np.min(x_all) - x_margin, np.max(x_all) + x_margin)
    ylim = (np.max(y_all) + y_margin, np.min(y_all) - y_margin)  # inverted

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    for ax, (c, m, label) in zip(axes, panels):
        ax.scatter(c, m, s=14, color='crimson')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'{label} ({len(c)})')
        ax.set_xlabel(COLOR_LABEL)

    axes[0].set_ylabel(MAG_LABEL)
    fig.suptitle(f'Cluster {cluster_id}: CMD comparison', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_three_panel_cmd(cluster_color, cluster_mag, member_mask,
                          cluster_id, out_path):
    """
    Three-panel, non-normalized CMD comparison, in the style of
    Ivanov et al. (2005) Fig. 4 (left/middle/right panels, shared axes,
    star counts given in brackets):
      1. All stars in the cluster area -- uncleaned, unfiltered.
      2. Cluster only -- the cleaned members that survived the
         membership-probability cut.
      3. Field only -- the stars removed as likely contamination.
    All three panels share the same x/y limits so they are directly
    comparable by eye, exactly as in the paper.
    """
    removed_mask = ~member_mask

    panels = [
        (cluster_color, cluster_mag, 'All stars'),
        (cluster_color[member_mask], cluster_mag[member_mask], 'Cluster'),
        (cluster_color[removed_mask], cluster_mag[removed_mask], 'Field'),
    ]

    # shared axis limits across all three panels, with a small margin
    x_all, y_all = cluster_color, cluster_mag
    x_margin = 0.05 * (np.max(x_all) - np.min(x_all) + 1e-9)
    y_margin = 0.05 * (np.max(y_all) - np.min(y_all) + 1e-9)
    xlim = (np.min(x_all) - x_margin, np.max(x_all) + x_margin)
    ylim = (np.max(y_all) + y_margin, np.min(y_all) - y_margin)  # inverted

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

    for ax, (c, m, label) in zip(axes, panels):
        ax.scatter(c, m, s=14, color='crimson')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f'{label} ({len(c)})')
        ax.set_xlabel(COLOR_LABEL)

    axes[0].set_ylabel(MAG_LABEL)
    fig.suptitle(f'Cluster {cluster_id}: CMD comparison', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_member_field_cmd(cluster_color, cluster_mag, member_mask,
                           cluster_id, out_path):
    """
    Single CMD (raw/original color and mag) showing cluster members
    (passed the membership-probability cutoff) in red and field stars
    (removed as likely contamination) in blue, on the same axes.
    """
    removed_mask = ~member_mask

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(cluster_color[member_mask], cluster_mag[member_mask],
               s=18, color='red',
               label=f'cluster members ({member_mask.sum()})', zorder=2)
    ax.scatter(cluster_color[removed_mask], cluster_mag[removed_mask],
               s=18, color='blue',
               label=f'field stars ({removed_mask.sum()})', zorder=1)
    ax.invert_yaxis()
    ax.set_xlabel(COLOR_LABEL)
    ax.set_ylabel(MAG_LABEL)
    ax.set_title(f'Cluster {cluster_id}: members (red) vs field stars (blue)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_probability_colored_cmd(cluster_color, cluster_mag, probs,
                                  cluster_id, out_path):
    """
    Single CMD (raw/original color and mag) with every star in the
    cluster area -- both eventual members and eventual field stars --
    colored on a continuous scale by its membership probability
    (0 = certainly field, 1 = certainly cluster).
    """
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(cluster_color, cluster_mag, c=probs, cmap='RdYlBu',
                     vmin=0, vmax=1, s=22, edgecolor='none')
    ax.invert_yaxis()
    ax.set_xlabel(COLOR_LABEL)
    ax.set_ylabel(MAG_LABEL)
    ax.set_title(f'Cluster {cluster_id}: membership probability')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('membership probability')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_luminosity_function(cluster_mag, field_cmd_list, cluster_id,
                              out_path, bin_width=0.5, mag_filter_name='F814W'):
    """
    Luminosity function comparison, in the style of Ivanov et al. (2005)
    Fig. 2 -- two histograms of star counts vs. magnitude on the same
    axes, plotted as log N (star counts on a log scale) vs. magnitude
    in the given filter:
      1. ALL stars in the cluster area (field + cluster stars combined,
         i.e. the raw, unfiltered cluster-region star list -- this
         script does not separate cluster members from contaminants
         at the point this plot is made, matching the paper's "all
         stars in the field of the cluster" curve).
      2. ALL stars in the background/sky regions combined, with the
         counts divided by the number of regions used -- so this curve
         represents "star counts per one region-sized area", directly
         comparable to curve 1 since the cluster area and each
         background region were constructed to have the same size.

    A cluster area with real excess stars over background (particularly
    at bright magnitudes, e.g. a red clump or RGB) will show curve 1
    sitting above curve 2 at those magnitudes; where the two curves
    overlap, that magnitude range is dominated by ordinary field stars.
    """
    field_mags = [m for c, m in field_cmd_list if len(m) > 0]
    n_regions_used = len(field_mags)
    combined_field_mag = (np.concatenate(field_mags) if n_regions_used > 0
                           else np.array([]))

    all_values = (np.concatenate([cluster_mag, combined_field_mag])
                  if n_regions_used > 0 else cluster_mag)
    if len(all_values) == 0:
        return  # nothing to plot

    mn = np.floor(np.min(all_values) / bin_width) * bin_width
    mx = np.ceil(np.max(all_values) / bin_width) * bin_width
    bins = np.arange(mn, mx + bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    cluster_counts, _ = np.histogram(cluster_mag, bins=bins)
    field_counts_raw, _ = np.histogram(combined_field_mag, bins=bins)
    field_counts_norm = (field_counts_raw / n_regions_used
                          if n_regions_used > 0 else field_counts_raw)

    # Compute log10(N) by hand and plot on a plain LINEAR axis, instead
    # of using matplotlib's automatic log-scale (which prints cluttered
    # ticks like "6 x 10^0" for small integer counts). This matches the
    # paper's own convention (Fig. 3, "lg(F)") of clean, simple tick
    # values. Bins with zero stars have no defined log -- they are left
    # as gaps in the line rather than plotted at some arbitrary floor.
    with np.errstate(divide='ignore'):
        cluster_log = np.where(cluster_counts > 0,
                                np.log10(cluster_counts), np.nan)
        field_log = np.where(field_counts_norm > 0,
                              np.log10(field_counts_norm), np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(bin_centers, cluster_log, where='mid', color='crimson',
             linewidth=1.5,
             label=f'cluster area, all stars (N={len(cluster_mag)})')
    ax.step(bin_centers, field_log, where='mid', color='steelblue',
             linewidth=1.5, linestyle='--',
             label=f'background area, avg of {n_regions_used} regions '
                    f'(N={len(combined_field_mag)} total)')
    ax.set_xlabel(f'{mag_filter_name} mag')
    ax.set_ylabel('log N')
    ax.set_title(f'Cluster {cluster_id}: luminosity function')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# Process ONE cluster, fully, with the chosen normalization method.
# ---------------------------------------------------------------
def process_one_cluster(cluster_id, color, mag, cluster_label,
                         bg_cluster_id, bg_region_index, method,
                         prob_cutoff, out_dir, ra, dec, X, Y):
    cluster_mask = (cluster_label == cluster_id)
    cluster_color = color[cluster_mask]
    cluster_mag = mag[cluster_mask]
    cluster_ra = ra[cluster_mask]
    cluster_dec = dec[cluster_mask]
    cluster_X = X[cluster_mask]
    cluster_Y = Y[cluster_mask]

    cluster_dir = os.path.join(out_dir, f'cluster_{cluster_id:02d}')
    os.makedirs(cluster_dir, exist_ok=True)

    # Output (extra 1): raw (non-normalized) CMD of the unfiltered cluster
    raw_png = os.path.join(cluster_dir, f'raw_cmd_cluster_{cluster_id}.png')
    plot_raw_cmd(cluster_color, cluster_mag, cluster_id, raw_png,
                 title_suffix=' (unfiltered)')
    print(f"  Saved: {raw_png}")

    # --- normalize the cluster's own CMD (scale derived from the cluster) ---
    color_scale, mag_scale, color_offset, mag_offset = get_normalization_scales(
        cluster_color, cluster_mag, method=method
    )
    cluster_norm = normalize_cmd(cluster_color, cluster_mag,
                                  color_scale, mag_scale,
                                  color_offset, mag_offset)
    color_n, mag_n = cluster_norm[:, 0], cluster_norm[:, 1]

    # Output (1): normalized CMD for the cluster
    norm_png = os.path.join(cluster_dir, f'normalized_cmd_cluster_{cluster_id}.png')
    plot_normalized_cmd(color_n, mag_n, cluster_id, method, norm_png)
    print(f"  Saved: {norm_png}")

    # --- Output (2): cluster CMD + each background region, one PNG each ---
    this_cluster_bg = (bg_cluster_id == cluster_id)
    region_ids = sorted(set(bg_region_index[this_cluster_bg]))

    field_cmd_list = []  # for the membership-probability step below
    for ri in region_ids:
        region_mask = this_cluster_bg & (bg_region_index == ri)
        region_color = color[region_mask]
        region_mag = mag[region_mask]
        field_cmd_list.append((region_color, region_mag))

        if len(region_color) == 0:
            continue

        region_norm = normalize_cmd(region_color, region_mag,
                                     color_scale, mag_scale,
                                     color_offset, mag_offset)
        region_png = os.path.join(
            cluster_dir, f'region_{ri:02d}_cmd_cluster_{cluster_id}.png'
        )
        plot_cluster_plus_region(color_n, mag_n,
                                  region_norm[:, 0], region_norm[:, 1],
                                  cluster_id, ri, method, region_png)
        print(f"  Saved: {region_png}")

    # --- Output (extra): luminosity function, cluster area vs.
    #     background area (normalized per region) ---
    lf_png = os.path.join(cluster_dir, f'luminosity_function_cluster_{cluster_id}.png')
    plot_luminosity_function(cluster_mag, field_cmd_list, cluster_id, lf_png)
    print(f"  Saved: {lf_png}")

    # --- Output (3): membership probabilities + cleaned CMD ---
    probs = compute_membership_probabilities(
        cluster_color, cluster_mag, field_cmd_list, norm_method=method
    )
    member_mask = probs >= prob_cutoff

    cleaned_png = os.path.join(cluster_dir, f'cleaned_cmd_cluster_{cluster_id}.png')
    plot_cleaned_cmd(color_n, mag_n, member_mask, cluster_id, method,
                      cleaned_png)
    print(f"  Saved: {cleaned_png}")

    # Output (extra 2): raw (non-normalized) CMD of the cleaned cluster
    raw_cleaned_png = os.path.join(
        cluster_dir, f'raw_cleaned_cmd_cluster_{cluster_id}.png'
    )
    plot_raw_cmd(cluster_color[member_mask], cluster_mag[member_mask],
                 cluster_id, raw_cleaned_png,
                 title_suffix=f' (cleaned, {member_mask.sum()}/'
                              f'{len(member_mask)} kept)')
    print(f"  Saved: {raw_cleaned_png}")

    # Output (extra 3): raw (non-normalized) CMD of the REMOVED stars --
    # the stars identified as likely field-star contamination and
    # dropped from the cluster by the membership-probability cut.
    removed_mask = ~member_mask
    raw_removed_png = os.path.join(
        cluster_dir, f'raw_removed_cmd_cluster_{cluster_id}.png'
    )
    plot_raw_cmd(cluster_color[removed_mask], cluster_mag[removed_mask],
                 cluster_id, raw_removed_png,
                 title_suffix=f' (removed field stars, '
                              f'{removed_mask.sum()}/{len(member_mask)})')
    print(f"  Saved: {raw_removed_png}")

    # Output (extra 4): three-panel CMD comparison (paper Fig. 4 style) --
    # all stars / cluster only / field only, side by side on shared axes.
    three_panel_png = os.path.join(
        cluster_dir, f'three_panel_cmd_cluster_{cluster_id}.png'
    )
    plot_three_panel_cmd(cluster_color, cluster_mag, member_mask,
                          cluster_id, three_panel_png)
    print(f"  Saved: {three_panel_png}")

    # Output (extra 5): single CMD, cluster members (red) + field stars
    # (blue) together on the same raw-value axes.
    member_field_png = os.path.join(
        cluster_dir, f'member_field_cmd_cluster_{cluster_id}.png'
    )
    plot_member_field_cmd(cluster_color, cluster_mag, member_mask,
                           cluster_id, member_field_png)
    print(f"  Saved: {member_field_png}")

    # Output (extra 6): single CMD, every star (cluster + field) colored
    # by its continuous membership probability.
    probability_cmd_png = os.path.join(
        cluster_dir, f'probability_cmd_cluster_{cluster_id}.png'
    )
    plot_probability_colored_cmd(cluster_color, cluster_mag, probs,
                                  cluster_id, probability_cmd_png)
    print(f"  Saved: {probability_cmd_png}")

    # bonus: per-star membership table for this cluster
    member_table = Table({
        'ra_deg': cluster_ra,
        'dec_deg': cluster_dec,
        'X_arcmin': cluster_X,
        'Y_arcmin': cluster_Y,
        'color': cluster_color,
        'mag': cluster_mag,
        'color_normalized': color_n,
        'mag_normalized': mag_n,
        'membership_probability': probs,
        'is_member': member_mask,
    })
    csv_path = os.path.join(cluster_dir, f'membership_cluster_{cluster_id}.csv')
    member_table.write(csv_path, format='csv', overwrite=True)
    print(f"  Saved: {csv_path}")

    n_raw = len(cluster_color)
    n_kept = member_mask.sum()
    print(f"\n  Cluster {cluster_id} summary: {n_raw} raw members -> "
          f"{n_kept} cleaned members ({n_kept/n_raw*100:.0f}% kept), "
          f"using {len(region_ids)} background regions, method='{method}'")


# ---------------------------------------------------------------
# Main interactive driver
# ---------------------------------------------------------------
def run_interactive():
    fits_path = input(
        "Enter the path to background_regions_full.fits "
        "(from select_background_regions.py): "
    ).strip().strip('"').strip("'")
    fits_path = os.path.expanduser(fits_path)

    if not os.path.isfile(fits_path):
        print(f"File not found: {fits_path}")
        sys.exit(1)

    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        required = {'color', 'mag', 'cluster_label', 'bg_cluster_id',
                    'bg_region_index', 'ra', 'dec', 'X', 'Y'}
        missing = required - set(data.columns.names)
        if missing:
            print(f"Input FITS file is missing required column(s): {missing}. "
                  f"Available columns: {data.columns.names}. "
                  f"Did you pass background_regions_full.fits (not "
                  f"background_regions.fits)?")
            sys.exit(1)

        color = np.array(data['color'], dtype=float)
        mag = np.array(data['mag'], dtype=float)
        cluster_label = np.array(data['cluster_label'], dtype=int)
        bg_cluster_id = np.array(data['bg_cluster_id'], dtype=int)
        bg_region_index = np.array(data['bg_region_index'], dtype=int)
        ra = np.array(data['ra'], dtype=float)
        dec = np.array(data['dec'], dtype=float)
        X = np.array(data['X'], dtype=float)
        Y = np.array(data['Y'], dtype=float)

    cluster_ids = sorted(set(cluster_label[cluster_label != -1]))
    if not cluster_ids:
        print("No clusters (cluster_label >= 0) found in this file.")
        sys.exit(1)

    print(f"\nFound {len(cluster_ids)} clusters: {cluster_ids}\n")

    cutoff_input = input(
        "Membership probability cutoff to keep a star [default 0.5]: "
    ).strip()
    prob_cutoff = float(cutoff_input) if cutoff_input else 0.5

    out_dir = os.path.dirname(os.path.abspath(fits_path))

    idx = 0
    processed = []
    while idx < len(cluster_ids):
        cluster_id = cluster_ids[idx]
        n_stars = int((cluster_label == cluster_id).sum())

        print(f"\n{'='*60}")
        print(f"Cluster {cluster_id} ({n_stars} raw member stars)")
        print(f"{'='*60}")

        proceed = ask_yes_no(
            f"Proceed with normalizing and cleaning cluster {cluster_id}?"
        )
        if not proceed:
            print(f"  Skipped cluster {cluster_id}.")
        else:
            method = ask_normalization_method()
            process_one_cluster(cluster_id, color, mag, cluster_label,
                                 bg_cluster_id, bg_region_index, method,
                                 prob_cutoff, out_dir, ra, dec, X, Y)
            processed.append(cluster_id)

        idx += 1
        if idx >= len(cluster_ids):
            break

        next_id = cluster_ids[idx]
        nav = input(
            f"\nContinue to next cluster ({next_id})? "
            f"[Enter = yes / cluster ID to jump to it / q to quit]: "
        ).strip().lower()

        if nav == 'q':
            break
        elif nav == '':
            continue
        else:
            try:
                jump_id = int(nav)
                if jump_id in cluster_ids:
                    idx = cluster_ids.index(jump_id)
                else:
                    print(f"  Cluster {jump_id} not found -- continuing "
                          f"to next cluster in order instead.")
            except ValueError:
                print("  Not recognized -- continuing to next cluster "
                      "in order.")

    print(f"\n{'='*60}")
    print(f"Done. Processed clusters: {processed}")
    print(f"All outputs saved under: {out_dir} (one subfolder per cluster)")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_interactive()
