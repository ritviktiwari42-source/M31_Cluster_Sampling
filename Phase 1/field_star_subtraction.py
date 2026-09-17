"""
field_star_subtraction.py

Field-star contamination removal for a DBSCAN-identified cluster,
using CMD (color-magnitude diagram) nearest-neighbor matching against
multiple sky background regions.

WORKFLOW:
  1. Build the CMD (color, magnitude) for one DBSCAN cluster's stars.
  2. Normalize the CMD distance metric (three options -- see below).
  3. Select ~20 sky background regions (empty of the cluster).
  4. For each background region, plot its stars on the SAME
     (normalized) cluster CMD space, and find each background star's
     nearest neighbor among the cluster stars -> flag that cluster
     star as a likely contaminant for this region.
  5. Repeat independently for all ~20 background regions.
  6. A cluster star's membership probability = fraction of background
     regions in which it was NOT flagged as someone's nearest match.
  7. Keep stars above your chosen probability cutoff (e.g. >=0.5).

DISTANCE METRICS (choose via `norm_method`):
  'simple'      : Dist = sqrt( (Color_i-Color_j)^2 + (Mag_i-Mag_j)^2 )
  'minmax'      : normalize each axis by (max-min) of the CLUSTER's
                   own color/mag range before computing distance
  'percentile'  : normalize each axis by the (95th-5th) percentile
                   range of the CLUSTER's own color/mag distribution
                   (more robust to outliers than 'minmax')

Normalization statistics (min/max or 5th/95th percentiles) are always
computed from the CLUSTER's stars, then applied identically to both
the cluster stars and every background region's stars -- so everyone
is being measured on the same ruler.
"""

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------
# 1. Normalization: compute scale factors AND an offset from the
#    cluster CMD, then apply both to any CMD (cluster or background
#    field). Subtracting the offset before dividing centers the data
#    near zero -- e.g. for 'percentile', a star at the cluster's 5th
#    percentile maps to ~0 and a star at the 95th percentile maps to
#    ~1, instead of both retaining their full raw magnitude (~25) and
#    only being divided by a small range (~3), which is what produced
#    the inflated 7-8 range in the un-centered version.
#
#    IMPORTANT: subtracting a constant offset does NOT change any
#    pairwise distance used in the nearest-neighbor matching --
#    (a-offset) - (b-offset) = a-b, so this is a pure display/
#    interpretability fix and never alters membership results.
# ---------------------------------------------------------------
def get_normalization_scales(color, mag, method='percentile'):
    """
    color, mag : 1D arrays -- the CLUSTER's color and magnitude values
                 (normalization stats are always derived from the
                 cluster, per the method description)
    method     : 'simple', 'minmax', or 'percentile'

    Returns
    -------
    color_scale, mag_scale, color_offset, mag_offset : floats.
    Normalized value = (raw - offset) / scale.
    For 'simple', scale is 1.0 (no rescaling) but the offset still
    centers the data (median) for readable plots; distances are
    unaffected either way.
    """
    if method == 'simple':
        color_offset = np.median(color)
        mag_offset = np.median(mag)
        return 1.0, 1.0, color_offset, mag_offset

    elif method == 'minmax':
        color_offset = np.min(color)
        mag_offset = np.min(mag)
        color_scale = np.max(color) - color_offset
        mag_scale = np.max(mag) - mag_offset

    elif method == 'percentile':
        c_lo, c_hi = np.percentile(color, [5, 95])
        m_lo, m_hi = np.percentile(mag, [5, 95])
        color_offset = c_lo
        mag_offset = m_lo
        color_scale = c_hi - c_lo
        mag_scale = m_hi - m_lo

    else:
        raise ValueError(
            f"Unknown norm_method '{method}'. "
            "Choose 'simple', 'minmax', or 'percentile'."
        )

    # guard against a zero-width range (e.g. a degenerate cluster)
    color_scale = color_scale if color_scale > 0 else 1.0
    mag_scale = mag_scale if mag_scale > 0 else 1.0

    return color_scale, mag_scale, color_offset, mag_offset


def normalize_cmd(color, mag, color_scale, mag_scale,
                   color_offset=0.0, mag_offset=0.0):
    """Center and scale a CMD's color/mag axes so distances become
    comparable and values sit near zero (not offset by the raw
    magnitude scale)."""
    return np.column_stack([(color - color_offset) / color_scale,
                             (mag - mag_offset) / mag_scale])


# ---------------------------------------------------------------
# 2. Core matching step: for one background region, flag likely
#    field-star contaminants among the cluster stars.
# ---------------------------------------------------------------
def flag_contaminants_one_field(cluster_cmd_norm, field_cmd_norm):
    """
    cluster_cmd_norm : (N, 2) array -- normalized [color, mag] for
                        cluster stars
    field_cmd_norm    : (M, 2) array -- normalized [color, mag] for
                        one background region's stars

    For every background star, find its single nearest cluster star
    in (normalized) CMD space and flag that cluster star.

    Returns
    -------
    flagged : (N,) boolean array, True = flagged in this round
    """
    tree = cKDTree(cluster_cmd_norm)
    _, nearest_idx = tree.query(field_cmd_norm, k=1)

    flagged = np.zeros(len(cluster_cmd_norm), dtype=bool)
    flagged[nearest_idx] = True
    return flagged


# ---------------------------------------------------------------
# 3. Run the process across all background regions and compute
#    membership probability for every cluster star.
# ---------------------------------------------------------------
def compute_membership_probabilities(color, mag, field_cmd_list,
                                      norm_method='percentile'):
    """
    color, mag      : 1D arrays -- cluster stars' color and magnitude
    field_cmd_list  : list of (color, mag) tuples, one pair of 1D
                       arrays per background region
    norm_method     : 'simple', 'minmax', or 'percentile'

    Returns
    -------
    probs : (N,) array, membership probability per cluster star
            (fraction of background regions in which it was NOT
            flagged)
    """
    color_scale, mag_scale, color_offset, mag_offset = get_normalization_scales(
        color, mag, norm_method
    )
    cluster_cmd_norm = normalize_cmd(color, mag, color_scale, mag_scale,
                                      color_offset, mag_offset)

    n_stars = len(color)
    n_fields = len(field_cmd_list)
    survived_count = np.zeros(n_stars, dtype=int)

    for field_color, field_mag in field_cmd_list:
        if len(field_color) == 0:
            # empty background region contributes nothing -- skip
            n_fields -= 1
            continue
        field_cmd_norm = normalize_cmd(field_color, field_mag,
                                        color_scale, mag_scale,
                                        color_offset, mag_offset)
        flagged = flag_contaminants_one_field(cluster_cmd_norm, field_cmd_norm)
        survived_count += (~flagged).astype(int)

    if n_fields == 0:
        raise ValueError("All background regions were empty -- check "
                          "your background-region selection/geometry.")

    probs = survived_count / n_fields
    return probs


# ---------------------------------------------------------------
# 4. Helper: pull out N background regions near a cluster, offset in
#    position but same shape/size as the cluster region.
#    ADAPT THIS to your actual X, Y (or RA, Dec) geometry / survey
#    footprint -- e.g. avoid regions that fall outside your imaged
#    area or land on chip gaps.
# ---------------------------------------------------------------
def get_background_regions(X, Y, mag, color, cluster_center, cluster_radius,
                            n_fields=20, offset_factor=3.0, seed=None):
    """
    X, Y            : full-catalog position arrays (e.g. arcmin offsets)
    mag, color      : full-catalog photometry arrays, same length as X, Y
    cluster_center  : (x0, y0) of the cluster
    cluster_radius  : radius defining the cluster region (same units as X, Y)
    n_fields        : how many background regions to extract (default 20)
    offset_factor   : how many cluster-radii away to place field centers
    seed            : optional RNG seed for reproducible region placement

    Returns
    -------
    field_cmd_list : list of (color, mag) 1D-array tuples, one pair
                     per background region
    field_centers  : list of (fx, fy) tuples -- useful for plotting /
                     sanity-checking where each region actually landed
    """
    rng = np.random.default_rng(seed)
    x0, y0 = cluster_center
    r = cluster_radius

    # Distribute region centers on a circle around the cluster, with
    # a small random angular jitter so regions for different clusters
    # (or different runs) don't always land in identical spots.
    base_angles = np.linspace(0, 2 * np.pi, n_fields, endpoint=False)
    jitter = rng.uniform(-0.3, 0.3, size=n_fields) * (2 * np.pi / n_fields)
    angles = base_angles + jitter

    field_cmd_list = []
    field_centers = []

    for theta in angles:
        fx = x0 + offset_factor * r * np.cos(theta)
        fy = y0 + offset_factor * r * np.sin(theta)

        dist = np.sqrt((X - fx) ** 2 + (Y - fy) ** 2)
        in_field = dist <= r

        field_cmd_list.append((color[in_field], mag[in_field]))
        field_centers.append((fx, fy))

    return field_cmd_list, field_centers


# ---------------------------------------------------------------
# 5. Putting it together for ONE DBSCAN cluster.
# ---------------------------------------------------------------
def clean_one_cluster(X, Y, mag, color, cluster_mask,
                       n_fields=20, offset_factor=3.0, prob_cutoff=0.5,
                       norm_method='percentile', seed=None):
    """
    X, Y          : full-catalog positions (arcmin offsets, same as your
                    DBSCAN XY array)
    mag, color    : full-catalog photometry, aligned with X, Y
    cluster_mask  : boolean array, True for stars belonging to this
                    DBSCAN cluster (e.g. labels == k)
    norm_method   : 'simple', 'minmax', or 'percentile'

    Returns
    -------
    member_mask    : boolean array (same length as X), True for stars
                      that passed the >= prob_cutoff membership test
    probs          : membership probability for each star IN the
                      cluster (aligned with cluster_mask's True entries)
    field_centers  : list of (fx, fy) background region centers used
                      (handy for plotting a sanity-check map)
    """
    cx = X[cluster_mask]
    cy = Y[cluster_mask]
    cluster_center = (np.mean(cx), np.mean(cy))
    cluster_radius = np.max(np.sqrt((cx - cluster_center[0]) ** 2 +
                                     (cy - cluster_center[1]) ** 2))

    cluster_color = color[cluster_mask]
    cluster_mag = mag[cluster_mask]

    field_cmd_list, field_centers = get_background_regions(
        X, Y, mag, color, cluster_center, cluster_radius,
        n_fields=n_fields, offset_factor=offset_factor, seed=seed
    )

    probs = compute_membership_probabilities(
        cluster_color, cluster_mag, field_cmd_list, norm_method=norm_method
    )

    # Map back to full-catalog boolean mask
    member_mask = np.zeros(len(X), dtype=bool)
    idx_in_full = np.where(cluster_mask)[0]
    member_mask[idx_in_full[probs >= prob_cutoff]] = True

    return member_mask, probs, field_centers


# ---------------------------------------------------------------
# EXAMPLE USAGE (after your existing DBSCAN block):
# ---------------------------------------------------------------
if __name__ == "__main__":
    # --- Replace these with your real arrays ---
    # X, Y      : from your DBSCAN script (arcmin offsets)
    # labels    : from db.labels_ (DBSCAN cluster labels)
    # mag_full  : magnitude column from phast_subset.fits
    # color_full: color column (e.g. mag_filter1 - mag_filter2)
    #
    # mag_full   = phast_table['YOUR_MAG_COLUMN'][Cond]
    # color_full = phast_table['YOUR_COLOR_COLUMN'][Cond]
    #
    # for k in set(labels):
    #     if k == -1:
    #         continue  # skip noise
    #     cluster_mask = (labels == k)
    #     member_mask, probs, field_centers = clean_one_cluster(
    #         X, Y, mag_full, color_full, cluster_mask,
    #         n_fields=20, offset_factor=3.0, prob_cutoff=0.5,
    #         norm_method='percentile'
    #     )
    #     print(f"Cluster {k}: {cluster_mask.sum()} raw -> "
    #           f"{member_mask.sum()} cleaned members")
    pass
