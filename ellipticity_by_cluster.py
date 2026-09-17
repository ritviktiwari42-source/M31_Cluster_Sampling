"""
ellipticity_by_cluster.py

Adapted from ellipticity_01.py -- instead of fitting an artificial,
synthetically-generated Gaussian+background signal, this version:

  1. Asks for the FITS file produced by the DBSCAN clusterfinder code
     (M31_clusterfinder_101.py), which must contain at least X, Y
     (arcmin offsets) and cluster_label.
  2. Finds every cluster DBSCAN identified (cluster_label >= 0).
  3. For EACH cluster, asks your permission before running the
     ellipticity fit on it (you can skip any cluster you don't want
     processed).
  4. For a cluster you approve, defines a local analysis region around
     it (a box padded out from the cluster's own DBSCAN radius so
     there's genuine surrounding "background" to fit against), pulls
     in every star from the FITS file that falls in that box
     (regardless of DBSCAN label), and fits the SAME Gaussian +
     uniform-background mixture model as the original script --
     independently estimating the object's center, shape (a, b,
     angle), ellipticity, and signal fraction directly from the point
     positions, without relying on DBSCAN's own membership labels.
  5. Bootstraps for parameter errors, computes an independent
     background density from the area outside the 5-sigma ellipse,
     and reports signal/background ratios at 1/2/3 sigma -- exactly
     as the original script did.
  6. Saves, per cluster you approved:
       - a results text file (ellipse_results_cluster_<id>.txt)
       - a plot PNG (ellipse_fit_cluster_<id>.png)
       - a FITS file (ellipse_data_cluster_<id>.fits) containing every
         star in that cluster's analysis region (X, Y, cluster_label)
         plus the fitted ellipse geometry in its primary header. This
         FITS file is what king_profile_104.py expects as input, so
         you can run a King-profile fit on this exact same cluster
         and region without re-running DBSCAN or re-deriving the
         center by hand.

USAGE:
    python ellipticity_by_cluster.py
    (prompts for everything -- no command-line arguments needed)
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
from astropy.io import fits


# ============================================================
# Gaussian PDF (unchanged from ellipticity_01.py)
# ============================================================
def gaussian_pdf(data, mean, cov):
    diff = data - mean
    det = np.linalg.det(cov)
    if det <= 0:
        return np.full(len(data), 1e-300)
    inv_cov = np.linalg.inv(cov)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
    return norm * np.exp(exponent)


# ============================================================
# Gaussian + uniform background fit (unchanged from
# ellipticity_01.py, except x_min/x_max/y_min/y_max are now
# parameters instead of module-level globals, since each cluster
# gets its own analysis region).
# ============================================================
def fit_gaussian_background(x, y, x_min, x_max, y_min, y_max):
    data = np.column_stack((x, y))
    area = (x_max - x_min) * (y_max - y_min)
    uniform_pdf = 1.0 / area

    mean0 = np.mean(data, axis=0)
    cov0 = np.cov(data, rowvar=False)
    sigma_x0 = np.sqrt(cov0[0, 0])
    sigma_y0 = np.sqrt(cov0[1, 1])
    rho0 = cov0[0, 1] / (sigma_x0 * sigma_y0)
    rho0 = np.clip(rho0, -0.95, 0.95)
    f0 = 0.5

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    initial = np.array([
        mean0[0], mean0[1],
        np.log(sigma_x0), np.log(sigma_y0),
        np.arctanh(rho0), np.log(f0 / (1.0 - f0))
    ])

    def negative_log_likelihood(params):
        mux, muy, log_sx, log_sy, rho_raw, logit_f = params
        sx = np.exp(log_sx)
        sy = np.exp(log_sy)
        rho = np.tanh(rho_raw)
        f = sigmoid(logit_f)
        cov = np.array([[sx**2, rho*sx*sy], [rho*sx*sy, sy**2]])
        signal_pdf = gaussian_pdf(data, np.array([mux, muy]), cov)
        total_pdf = f * signal_pdf + (1.0 - f) * uniform_pdf
        total_pdf = np.maximum(total_pdf, 1e-300)
        return -np.sum(np.log(total_pdf))

    result = minimize(
        negative_log_likelihood, initial, method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-7}
    )
    if not result.success:
        print("  Warning: fit did not fully converge.")

    mux, muy, log_sx, log_sy, rho_raw, logit_f = result.x
    sx = np.exp(log_sx)
    sy = np.exp(log_sy)
    rho = np.tanh(rho_raw)
    f = sigmoid(logit_f)
    cov = np.array([[sx**2, rho*sx*sy], [rho*sx*sy, sy**2]])

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    a = np.sqrt(eigenvalues[0])
    b = np.sqrt(eigenvalues[1])
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    ellipticity = 1.0 - b / a

    return np.array([mux, muy, a, b, angle, ellipticity, f])


def bootstrap_fit(x, y, x_min, x_max, y_min, y_max, n_bootstrap=50):
    data = np.column_stack((x, y))
    bootstrap_results = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, len(data), size=len(data))
        sample = data[indices]
        try:
            result = fit_gaussian_background(
                sample[:, 0], sample[:, 1], x_min, x_max, y_min, y_max
            )
            if result is not None and np.all(np.isfinite(result)):
                bootstrap_results.append(result)
        except Exception:
            pass
    bootstrap_results = np.array(bootstrap_results)
    if len(bootstrap_results) < 2:
        raise RuntimeError("Not enough successful bootstrap fits.")
    return np.std(bootstrap_results, axis=0, ddof=1)


def ellipse_radius_squared(data, center, a, b, angle_deg):
    angle_rad = np.radians(angle_deg)
    dx = data[:, 0] - center[0]
    dy = data[:, 1] - center[1]
    xp = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
    yp = -dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    return (xp / a) ** 2 + (yp / b) ** 2


def ellipse_area_inside_rectangle(center, a, b, angle_deg, x_min, x_max,
                                    y_min, y_max, n_grid=2000):
    x_grid = np.linspace(x_min, x_max, n_grid)
    y_grid = np.linspace(y_min, y_max, n_grid)
    dx_grid = x_grid[1] - x_grid[0]
    dy_grid = y_grid[1] - y_grid[0]
    area_inside = 0.0
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    chunk_size = 200
    for i in range(0, n_grid, chunk_size):
        x_chunk = x_grid[i:i + chunk_size]
        xx, yy = np.meshgrid(x_chunk, y_grid)
        dx = xx - center[0]
        dy = yy - center[1]
        xp = dx * cos_a + dy * sin_a
        yp = -dx * sin_a + dy * cos_a
        inside = (xp / a) ** 2 + (yp / b) ** 2 <= 1.0
        area_inside += np.sum(inside)
    area_inside *= dx_grid * dy_grid
    return area_inside


# ============================================================
# NEW: write a per-cluster FITS file that king_profile_104.py
# can read directly. Primary header carries the fitted ellipse
# geometry (used by the King code as the profile center and, by
# default, as the maximum fit radius); the binary table carries
# every star in the analysis region, tagged with its DBSCAN label
# so downstream code can still tell members from field stars if
# it wants to.
# ============================================================
def save_cluster_fits(cluster_id, x, y, region_cluster_label, center, a, b,
                       angle, ellipticity, signal_fraction, radius,
                       half_width, x_min, x_max, y_min, y_max, out_dir,
                       extra_cols=None, tile_meta=None, fit_errors=None,
                       n_dbscan_members=None):
    col_x = fits.Column(name='X', format='D', array=np.asarray(x, dtype=float))
    col_y = fits.Column(name='Y', format='D', array=np.asarray(y, dtype=float))
    col_label = fits.Column(name='cluster_label', format='K',
                             array=np.asarray(region_cluster_label, dtype=np.int64))
    cols = [col_x, col_y, col_label]

    # Carry ra/dec and BOTH filters' magnitudes+errors straight through,
    # so nothing downstream has to go back to the DBSCAN export for them.
    if extra_cols:
        for name, arr in extra_cols.items():
            cols.append(fits.Column(name=name, format='D',
                                     array=np.asarray(arr, dtype=float)))

    table_hdu = fits.BinTableHDU.from_columns(cols, name='CLUSTER_DATA')

    hdr = fits.Header()
    hdr['CLUSTID'] = (int(cluster_id), 'DBSCAN cluster ID')
    hdr['CENTX'] = (float(center[0]), 'Fitted ellipse center X, arcmin')
    hdr['CENTY'] = (float(center[1]), 'Fitted ellipse center Y, arcmin')
    hdr['FIT_A'] = (float(a), 'Fitted semi-major axis, arcmin')
    hdr['FIT_B'] = (float(b), 'Fitted semi-minor axis, arcmin')
    hdr['ANGLE'] = (float(angle), 'Fitted ellipse position angle, deg')
    hdr['ELLIP'] = (float(ellipticity), 'Fitted ellipticity, 1 - b/a')
    hdr['SIGFRAC'] = (float(signal_fraction), 'Fitted Gaussian signal fraction')
    hdr['DBRADIUS'] = (float(radius), 'DBSCAN cluster radius, arcmin')
    hdr['HALFWID'] = (float(half_width),
                       'Analysis box half-width, arcmin')
    hdr['XMIN'] = (float(x_min), 'Analysis box X min, arcmin')
    hdr['XMAX'] = (float(x_max), 'Analysis box X max, arcmin')
    hdr['YMIN'] = (float(y_min), 'Analysis box Y min, arcmin')
    hdr['YMAX'] = (float(y_max), 'Analysis box Y max, arcmin')
    hdr['NPOINTS'] = (len(x), 'Number of stars in analysis region')

    # Bootstrap errors: these were already being computed, but were not
    # reaching the FITS file, so the final CSV had no ellipse errors.
    if fit_errors is not None:
        hdr['CENTXERR'] = (float(fit_errors[0]), 'Error on CENTX, arcmin')
        hdr['CENTYERR'] = (float(fit_errors[1]), 'Error on CENTY, arcmin')
        hdr['FIT_AERR'] = (float(fit_errors[2]), 'Error on FIT_A, arcmin')
        hdr['FIT_BERR'] = (float(fit_errors[3]), 'Error on FIT_B, arcmin')
        hdr['ANGLEERR'] = (float(fit_errors[4]), 'Error on ANGLE, deg')
        hdr['ELLIPERR'] = (float(fit_errors[5]), 'Error on ELLIP')
        hdr['SIGFRERR'] = (float(fit_errors[6]), 'Error on SIGFRAC')

    if n_dbscan_members is not None:
        hdr['NDBSCAN'] = (int(n_dbscan_members),
                           'Stars DBSCAN assigned to this cluster')

    # Tile reference point, so the cluster center can be converted back
    # to RA/Dec anywhere downstream without re-deriving it.
    if tile_meta:
        for key, value in tile_meta.items():
            hdr[key] = value

    hdr['ORIGIN'] = ('ellipticity_by_cluster.py', 'Producing script')

    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([primary_hdu, table_hdu])

    fits_path = os.path.join(out_dir, f'ellipse_data_cluster_{cluster_id}.fits')
    hdul.writeto(fits_path, overwrite=True)
    return fits_path


# ============================================================
# Per-cluster driver: define the local region, run the fit,
# bootstrap, compute background/S-B, save results + plot + FITS.
# ============================================================
def process_one_cluster(cluster_id, X_all, Y_all, cluster_label,
                         out_dir, padding_factor, n_bootstrap,
                         extra_all=None, tile_meta=None):
    cluster_mask = (cluster_label == cluster_id)
    cx0 = np.mean(X_all[cluster_mask])
    cy0 = np.mean(Y_all[cluster_mask])
    radius = np.max(np.sqrt((X_all[cluster_mask] - cx0) ** 2 +
                              (Y_all[cluster_mask] - cy0) ** 2))

    half_width = radius * padding_factor
    x_min, x_max = cx0 - half_width, cx0 + half_width
    y_min, y_max = cy0 - half_width, cy0 + half_width

    in_region = ((X_all >= x_min) & (X_all <= x_max) &
                 (Y_all >= y_min) & (Y_all <= y_max))
    x = X_all[in_region]
    y = Y_all[in_region]
    region_cluster_label = cluster_label[in_region]
    n_points = len(x)

    print(f"  Cluster {cluster_id}: center=({cx0:.4f}, {cy0:.4f}), "
          f"DBSCAN radius={radius:.4f}, analysis box "
          f"half-width={half_width:.4f}, {n_points} stars in region")

    if n_points < 20:
        print(f"  Skipping cluster {cluster_id}: only {n_points} stars in "
              f"the analysis region -- too few for a reliable fit.")
        return

    fit = fit_gaussian_background(x, y, x_min, x_max, y_min, y_max)
    center = fit[0:2]
    a, b, angle, ellipticity, signal_fraction = fit[2], fit[3], fit[4], fit[5], fit[6]

    print(f"    Fitted: a={a:.4f}, b={b:.4f}, angle={angle:.2f} deg, "
          f"ellipticity={ellipticity:.4f}, signal_fraction={signal_fraction:.3f}")

    try:
        errors = bootstrap_fit(x, y, x_min, x_max, y_min, y_max,
                                n_bootstrap=n_bootstrap)
    except RuntimeError as e:
        print(f"    Bootstrap failed ({e}) -- skipping error estimate "
              f"for this cluster.")
        errors = np.full(7, np.nan)

    center_err = errors[0:2]
    a_err, b_err, angle_err = errors[2], errors[3], errors[4]
    ellipticity_err, signal_fraction_err = errors[5], errors[6]

    data = np.column_stack((x, y))
    r2 = ellipse_radius_squared(data, center, a, b, angle)

    total_area = (x_max - x_min) * (y_max - y_min)
    area_5sigma_inside = ellipse_area_inside_rectangle(
        center, 5.0 * a, 5.0 * b, angle, x_min, x_max, y_min, y_max
    )
    background_area = total_area - area_5sigma_inside
    if background_area <= 0:
        print(f"    Skipping background estimate for cluster {cluster_id}: "
              f"5-sigma ellipse covers the whole analysis region.")
        background_density = np.nan
    else:
        outside_5sigma = r2 > 5.0 ** 2
        N_outside_5sigma = np.sum(outside_5sigma)
        background_density = N_outside_5sigma / background_area

    sb_results = {}
    for n_sigma in [1, 2, 3]:
        ellipse_area = np.pi * (n_sigma * a) * (n_sigma * b)
        inside = r2 <= n_sigma ** 2
        N_observed = np.sum(inside)
        N_background_expected = (background_density * ellipse_area
                                  if np.isfinite(background_density) else np.nan)
        N_signal_estimated = (N_observed - N_background_expected
                               if np.isfinite(N_background_expected) else np.nan)
        S_over_B = (N_signal_estimated / N_background_expected
                    if (np.isfinite(N_background_expected) and
                        N_background_expected > 0) else np.nan)
        sb_results[n_sigma] = {
            'area': ellipse_area, 'N_observed': N_observed,
            'N_background': N_background_expected,
            'N_signal': N_signal_estimated, 'S_over_B': S_over_B,
        }

    # ---- Save results text file ----
    txt_path = os.path.join(out_dir, f'ellipse_results_cluster_{cluster_id}.txt')
    values = [
        center[0], center[1], a, b, angle, ellipticity, signal_fraction,
        center_err[0], center_err[1], a_err, b_err, angle_err,
        ellipticity_err, signal_fraction_err,
        total_area, area_5sigma_inside, background_area,
        int(np.sum(r2 > 5.0**2)) if np.isfinite(background_density) else -1,
        background_density,
    ]
    for n_sigma in [1, 2, 3]:
        r = sb_results[n_sigma]
        values.extend([r['area'], r['N_observed'], r['N_background'],
                        r['N_signal'], r['S_over_B']])
    with open(txt_path, 'w') as f:
        f.write(",".join(f"{v:.8g}" for v in values) + "\n")
    print(f"    Saved: {txt_path}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 8))
    is_member = (region_cluster_label == cluster_id)
    ax.scatter(x[~is_member], y[~is_member], s=8, alpha=0.35, color='gray',
               label='other stars in region')
    ax.scatter(x[is_member], y[is_member], s=10, alpha=0.6, color='crimson',
               label=f'DBSCAN cluster {cluster_id} members')
    ax.plot(center[0], center[1], marker='x', markersize=10,
            markeredgewidth=2, color='blue', label='fitted center')

    for n_sigma in [1, 2, 3]:
        ellipse = Ellipse(xy=center, width=2.0*n_sigma*a, height=2.0*n_sigma*b,
                           angle=angle, fill=False, linewidth=2,
                           label=f'{n_sigma}$\\sigma$')
        ax.add_patch(ellipse)

    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X, arcmin')
    ax.set_ylabel('Y, arcmin')
    ax.set_title(f'Cluster {cluster_id}: Gaussian + background fit\n'
                 f'Ellipticity = {ellipticity:.3f} +/- {ellipticity_err:.3f}')
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()

    png_path = os.path.join(out_dir, f'ellipse_fit_cluster_{cluster_id}.png')
    fig.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {png_path}")

    # ---- Save FITS file for use by king_profile_104.py ----
    fits_path = save_cluster_fits(
        cluster_id, x, y, region_cluster_label, center, a, b, angle,
        ellipticity, signal_fraction, radius, half_width,
        x_min, x_max, y_min, y_max, out_dir,
        extra_cols=({name: arr[in_region] for name, arr in extra_all.items()}
                    if extra_all else None),
        tile_meta=tile_meta,
        fit_errors=errors,
        n_dbscan_members=int((cluster_label == cluster_id).sum()),
    )
    print(f"    Saved: {fits_path}  (input for king_profile_104.py)")

    # Return a flat summary row so the caller can build the combined
    # per-tile FITS/CSV without re-opening any per-cluster file.
    tm = tile_meta or {}
    RAo = float(tm['RAO'][0]) if 'RAO' in tm else np.nan
    DECo = float(tm['DECO'][0]) if 'DECO' in tm else np.nan
    cos_DECo = float(tm['COSDECO'][0]) if 'COSDECO' in tm else np.nan
    if np.isfinite(RAo) and np.isfinite(DECo):
        cent_ra = RAo + center[0] / (cos_DECo * 60.0)
        cent_dec = DECo + center[1] / 60.0
    else:
        cent_ra = cent_dec = np.nan

    return {
        'cluster_id': int(cluster_id),
        'n_dbscan_members': int((cluster_label == cluster_id).sum()),
        'n_stars_in_region': int(len(x)),
        'dbscan_radius': float(radius),
        'halfwidth': float(half_width),
        'centx': float(center[0]),
        'centy': float(center[1]),
        'centx_err': float(center_err[0]),
        'centy_err': float(center_err[1]),
        'cent_ra': float(cent_ra),
        'cent_dec': float(cent_dec),
        'fit_a': float(a), 'fit_a_err': float(a_err),
        'fit_b': float(b), 'fit_b_err': float(b_err),
        'angle': float(angle), 'angle_err': float(angle_err),
        'ellipticity': float(ellipticity),
        'ellipticity_err': float(ellipticity_err),
        'signal_fraction': float(signal_fraction),
        'signal_fraction_err': float(signal_fraction_err),
        'fits_path': fits_path,
    }


# ============================================================
# Combined per-TILE outputs: one FITS + one CSV covering every
# cluster processed in this run. The per-cluster PNG/FITS files are
# still written as before -- these are additional, not replacements.
# ============================================================
ELLIPSE_SUMMARY_FIELDS = [
    'cluster_id', 'n_dbscan_members', 'n_stars_in_region',
    'dbscan_radius', 'halfwidth',
    'centx', 'centx_err', 'centy', 'centy_err', 'cent_ra', 'cent_dec',
    'fit_a', 'fit_a_err', 'fit_b', 'fit_b_err',
    'angle', 'angle_err', 'ellipticity', 'ellipticity_err',
    'signal_fraction', 'signal_fraction_err',
]


def save_combined_ellipse_outputs(rows, tile_meta, out_dir, input_name):
    """Writes ellipse_all_clusters.fits and ellipse_all_clusters.csv."""
    if not rows:
        return None, None

    int_fields = {'cluster_id', 'n_dbscan_members', 'n_stars_in_region'}
    cols = []
    for field in ELLIPSE_SUMMARY_FIELDS:
        values = [r[field] for r in rows]
        if field in int_fields:
            cols.append(fits.Column(name=field, format='K',
                                     array=np.array(values, dtype=np.int64)))
        else:
            cols.append(fits.Column(name=field, format='D',
                                     array=np.array(values, dtype=float)))
    table_hdu = fits.BinTableHDU.from_columns(cols, name='ELLIPSE_ALL')

    hdr = fits.Header()
    hdr['NCLUST'] = (len(rows), 'Clusters in this combined file')
    hdr['SRCFILE'] = (os.path.basename(input_name)[:60],
                       'DBSCAN export this came from')
    if tile_meta:
        for key, value in tile_meta.items():
            hdr[key] = value
    hdr['ORIGIN'] = ('ellipticity_by_cluster.py', 'Producing script')

    fits_path = os.path.join(out_dir, 'ellipse_all_clusters.fits')
    fits.HDUList([fits.PrimaryHDU(header=hdr), table_hdu]).writeto(
        fits_path, overwrite=True)

    csv_path = os.path.join(out_dir, 'ellipse_all_clusters.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ELLIPSE_SUMMARY_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in ELLIPSE_SUMMARY_FIELDS})

    return fits_path, csv_path


# ============================================================
# Main interactive driver
# ============================================================
def run_interactive():
    print("ellipticity_by_cluster.py -- Gaussian+background ellipticity "
          "fit, per DBSCAN cluster\n")

    path = input(
        "Enter the path to your DBSCAN-exported FITS file (e.g. "
        "dbscan_export_for_topcat.fits): "
    ).strip().strip('"').strip("'")
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    with fits.open(path) as hdul:
        data = hdul[1].data
        names = data.columns.names
        required = {'X', 'Y', 'cluster_label'}
        missing = required - set(names)
        if missing:
            print(f"Input file is missing required column(s): {missing}. "
                  f"Available columns: {names}")
            sys.exit(1)
        # Carry every extra column we can through to the per-cluster
        # FITS: sky coordinates and BOTH filters with their errors.
        # Anything absent is simply skipped rather than treated as fatal.
        extra_all = {}
        for col in ('ra', 'dec', 'color', 'mag',
                    'mag_F814W', 'mag_F814W_err',
                    'mag_F475W', 'mag_F475W_err'):
            if col in names:
                extra_all[col] = np.array(data[col], dtype=float)
        missing_extras = [c_ for c_ in ('ra', 'dec', 'mag_F814W', 'mag_F475W')
                          if c_ not in extra_all]
        if missing_extras:
            print(f"  Note: {missing_extras} not in this file -- they will "
                  f"not be carried into the per-cluster FITS files.")

        # Tile reference point, if the clusterfinder saved it.
        hdr_in = hdul[1].header
        tile_meta = {}
        if 'RAO' in hdr_in and 'DECO' in hdr_in:
            tile_meta['RAO'] = (float(hdr_in['RAO']),
                                 'Tile reference RA, deg')
            tile_meta['DECO'] = (float(hdr_in['DECO']),
                                  'Tile reference Dec, deg')
            tile_meta['COSDECO'] = (float(hdr_in.get(
                'COSDECO', np.cos(np.deg2rad(float(hdr_in['DECO']))))),
                'cos(DECO)')
            print(f"  Tile reference read from header: "
                  f"RAo={hdr_in['RAO']:.6f}, DECo={hdr_in['DECO']:.6f}")
        elif 'ra' in extra_all and 'dec' in extra_all:
            # Fall back to deriving it from the ra/dec + X/Y columns.
            _dec = extra_all['dec']; _ra = extra_all['ra']
            _Y = np.array(data['Y'], dtype=float)
            _X = np.array(data['X'], dtype=float)
            DECo_ = _dec[0] - _Y[0] / 60.0
            cos_ = np.cos(np.deg2rad(DECo_))
            RAo_ = _ra[0] - _X[0] / (cos_ * 60.0)
            tile_meta['RAO'] = (float(RAo_), 'Tile reference RA, deg')
            tile_meta['DECO'] = (float(DECo_), 'Tile reference Dec, deg')
            tile_meta['COSDECO'] = (float(cos_), 'cos(DECO)')
            print(f"  Tile reference derived from ra/dec + X/Y: "
                  f"RAo={RAo_:.6f}, DECo={DECo_:.6f}")
        else:
            print("  Warning: no tile reference (RAO/DECO) and no ra/dec "
                  "columns -- downstream RA/Dec conversion will not be "
                  "possible from these files.")

        X_all = np.array(data['X'], dtype=float)
        Y_all = np.array(data['Y'], dtype=float)
        cluster_label = np.array(data['cluster_label'], dtype=int)

    cluster_ids = sorted(c for c in set(cluster_label) if c >= 0)
    if not cluster_ids:
        print("No clusters (cluster_label >= 0) found in this file.")
        sys.exit(1)
    print(f"\nFound {len(cluster_ids)} clusters: {cluster_ids}")

    padding_input = input(
        "\nAnalysis region size, as a multiple of each cluster's own "
        "DBSCAN radius [default 8]: "
    ).strip()
    padding_factor = float(padding_input) if padding_input else 8.0

    bootstrap_input = input(
        "Number of bootstrap iterations for error estimates [default 50]: "
    ).strip()
    n_bootstrap = int(bootstrap_input) if bootstrap_input else 50

    out_dir = os.path.dirname(os.path.abspath(path))
    processed = []
    summary_rows = []

    for cluster_id in cluster_ids:
        n_members = int((cluster_label == cluster_id).sum())
        proceed = input(
            f"\nRun the ellipticity fit for cluster {cluster_id} "
            f"({n_members} DBSCAN members)? [Y/n]: "
        ).strip().lower()
        if proceed.startswith('n'):
            print(f"  Skipped cluster {cluster_id}.")
            continue

        summary_row = process_one_cluster(
            cluster_id, X_all, Y_all, cluster_label,
            out_dir, padding_factor, n_bootstrap,
            extra_all=extra_all, tile_meta=tile_meta)
        if summary_row is not None:
            summary_rows.append(summary_row)
        processed.append(cluster_id)

    combined_fits, combined_csv = save_combined_ellipse_outputs(
        summary_rows, tile_meta, out_dir, path)

    print(f"\nDone. Processed clusters: {processed}")
    print(f"All outputs saved to: {out_dir}")
    if combined_fits:
        print(f"\nCombined (whole-tile) outputs:")
        print(f"  {combined_fits}")
        print(f"  {combined_csv}")
        print("\nFeed the combined FITS straight into king_profile_104.py "
              "to fit every cluster in this tile in one go. The per-cluster "
              "'ellipse_data_cluster_<id>.fits' files are still written too, "
              "and are what the King step reads the actual star positions "
              "from.")


if __name__ == "__main__":
    run_interactive()
