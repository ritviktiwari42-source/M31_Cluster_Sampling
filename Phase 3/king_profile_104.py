"""
king_profile_104.py

Reworked so it no longer generates a synthetic catalogue. Instead it:

  1. Asks for the FITS file produced by ellipticity_by_cluster.py
     (an 'ellipse_data_cluster_<id>.fits' file), which carries the
     cluster's fitted center in its primary header (CENTX, CENTY) and
     the region's stars (X, Y, cluster_label) in a binary table.
  2. Builds the radial star-count profile around that center, exactly
     as your original king_profile_104.py did (annular bins, Poisson
     errors, non-empty-bin-only fit).
  3. Fits a King profile + constant background to the profile, with
     the same TRFLSQFitter machinery, error propagation, and
     concentration calculation as your original script.
  4. Prints the same style of results block as your original script.
  5. Saves a plot (PNG) and, for that cluster, a FITS file with the
     fitted parameters in its header and the radial profile + model
     curves in a binary table -- one FITS file per cluster you run
     this on.

USAGE:
    python king_profile_104.py
    (prompts for everything -- no command-line arguments needed)
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.modeling import models, fitting


# ============================================================
# 1. Ask for the cluster FITS file produced by
#    ellipticity_by_cluster.py
# ============================================================
def load_cluster_fits(path):
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        table = hdul[1].data

        required_header_keys = {'CENTX', 'CENTY'}
        missing_header = required_header_keys - set(hdr.keys())
        if missing_header:
            print(f"Input FITS file is missing required header keyword(s): "
                  f"{missing_header}. This doesn't look like an "
                  f"'ellipse_data_cluster_<id>.fits' file produced by "
                  f"ellipticity_by_cluster.py.")
            sys.exit(1)

        names = table.columns.names
        required_cols = {'X', 'Y'}
        missing_cols = required_cols - set(names)
        if missing_cols:
            print(f"Input FITS file is missing required column(s): "
                  f"{missing_cols}. Available columns: {names}")
            sys.exit(1)

        x0 = float(hdr['CENTX'])
        y0 = float(hdr['CENTY'])
        cluster_id = hdr.get('CLUSTID', 'unknown')
        half_width = hdr.get('HALFWID', None)

        # Everything the ellipticity step already knows, carried forward
        # verbatim so the final CSV never has to reopen earlier files.
        passthrough = {}
        for key in ('RAO', 'DECO', 'COSDECO', 'NDBSCAN', 'DBRADIUS',
                    'FIT_A', 'FIT_B', 'ANGLE', 'ELLIP', 'SIGFRAC',
                    'CENTXERR', 'CENTYERR', 'FIT_AERR', 'FIT_BERR',
                    'ANGLEERR', 'ELLIPERR', 'SIGFRERR',
                    'NPOINTS', 'XMIN', 'XMAX', 'YMIN', 'YMAX'):
            if key in hdr:
                passthrough[key] = hdr[key]

        x = np.array(table['X'], dtype=float)
        y = np.array(table['Y'], dtype=float)
        if 'cluster_label' in names:
            cluster_label = np.array(table['cluster_label'], dtype=int)
        else:
            cluster_label = np.full(len(x), -1, dtype=int)

    return x, y, x0, y0, cluster_id, half_width, cluster_label, passthrough


# ============================================================
# 2. Radial profile + King fit for one cluster
# ============================================================
def fit_king_profile(x, y, x0, y0, r_max, n_bins):

    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)

    # Only stars inside r_max are used -- for stars drawn from a
    # (roughly) square analysis box, r_max should not exceed the
    # box's half-width, or the outer annuli will be missing area
    # that falls outside the box and the density will be biased low.
    in_range = r <= r_max
    r = r[in_range]

    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_mid = 0.5 * (r_edges[:-1] + r_edges[1:])

    counts, _ = np.histogram(r, bins=r_edges)

    area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2)

    density = counts / area
    density_err = np.sqrt(counts) / area
    density_err[counts == 0] = 1.0 / area[counts == 0]

    good = counts > 0
    r_fit = r_mid[good]
    density_fit = density[good]
    density_err_fit = density_err[good]

    if len(r_fit) < 5:
        raise RuntimeError(
            f"Only {len(r_fit)} non-empty radial bins -- too few to fit a "
            f"King profile. Try a larger r_max, fewer bins, or check the "
            f"input file."
        )

    king_init = models.KingProjectedAnalytic1D(
        amplitude=np.max(density_fit),
        r_core=r_max / 8.0,
        r_tide=r_max,
    )
    background_init = models.Const1D(
        amplitude=np.median(density_fit[-5:])
    )
    model_init = king_init + background_init

    fitter = fitting.TRFLSQFitter()
    fit_model = fitter(
        model_init,
        r_fit,
        density_fit,
        weights=1.0 / density_err_fit,
    )

    amplitude = fit_model.amplitude_0.value
    rc = fit_model.r_core_0.value
    rt = fit_model.r_tide_0.value
    background = fit_model.amplitude_1.value

    cov = fitter.fit_info.get("param_cov", None)

    if cov is not None:
        errors = np.sqrt(np.diag(cov))
        amplitude_err, rc_err, rt_err, background_err = errors
    else:
        amplitude_err = rc_err = rt_err = background_err = np.nan

    concentration = np.log10(rt / rc)

    if cov is not None:
        dc_drc = -1.0 / (rc * np.log(10))
        dc_drt = 1.0 / (rt * np.log(10))
        concentration_err = np.sqrt(
            dc_drc ** 2 * cov[1, 1]
            + dc_drt ** 2 * cov[2, 2]
            + 2 * dc_drc * dc_drt * cov[1, 2]
        )
    else:
        concentration_err = np.nan

    results = {
        'r_edges': r_edges, 'r_mid': r_mid, 'counts': counts, 'area': area,
        'density': density, 'density_err': density_err, 'good': good,
        'r_fit': r_fit, 'density_fit': density_fit,
        'density_err_fit': density_err_fit,
        'fit_model': fit_model, 'cov': cov,
        'amplitude': amplitude, 'amplitude_err': amplitude_err,
        'rc': rc, 'rc_err': rc_err,
        'rt': rt, 'rt_err': rt_err,
        'background': background, 'background_err': background_err,
        'concentration': concentration, 'concentration_err': concentration_err,
    }
    return results


# ============================================================
# 3. Save a FITS file with the King fit results for this cluster
# ============================================================
def save_king_fits(cluster_id, results, x0, y0, r_max, n_bins, out_dir,
                    passthrough=None):
    col_r = fits.Column(name='R_MID', format='D', array=results['r_mid'])
    col_counts = fits.Column(name='N_COUNTS', format='K',
                              array=results['counts'].astype(np.int64))
    col_area = fits.Column(name='AREA', format='D', array=results['area'])
    col_density = fits.Column(name='DENSITY', format='D',
                               array=results['density'])
    col_density_err = fits.Column(name='DENSITY_ERR', format='D',
                                   array=results['density_err'])
    col_used = fits.Column(name='USED_IN_FIT', format='L',
                            array=results['good'])

    fit_model = results['fit_model']
    model_king = fit_model[0](results['r_mid'])
    model_bg = fit_model[1](results['r_mid'])
    model_total = fit_model(results['r_mid'])
    col_model_total = fits.Column(name='MODEL_TOTAL', format='D',
                                   array=model_total)
    col_model_king = fits.Column(name='MODEL_KING', format='D',
                                  array=model_king)
    col_model_bg = fits.Column(name='MODEL_BG', format='D', array=model_bg)

    table_hdu = fits.BinTableHDU.from_columns([
        col_r, col_counts, col_area, col_density, col_density_err,
        col_used, col_model_total, col_model_king, col_model_bg,
    ], name='KING_PROFILE')

    hdr = fits.Header()
    hdr['CLUSTID'] = (str(cluster_id), 'DBSCAN cluster ID')
    hdr['CENTX'] = (float(x0), 'Profile center X, arcmin')
    hdr['CENTY'] = (float(y0), 'Profile center Y, arcmin')
    hdr['RMAX'] = (float(r_max), 'Maximum radius used for the fit, arcmin')
    hdr['NBINS'] = (int(n_bins), 'Number of radial bins')
    hdr['AMPLITUD'] = (float(results['amplitude']), 'King amplitude')
    hdr['AMP_ERR'] = (float(results['amplitude_err']), 'King amplitude error')
    hdr['RCORE'] = (float(results['rc']), 'Core radius, arcmin')
    hdr['RCORE_ER'] = (float(results['rc_err']), 'Core radius error, arcmin')
    hdr['RTIDE'] = (float(results['rt']), 'Tidal radius, arcmin')
    hdr['RTIDE_ER'] = (float(results['rt_err']), 'Tidal radius error, arcmin')
    hdr['BACKGRND'] = (float(results['background']), 'Constant background')
    hdr['BACK_ERR'] = (float(results['background_err']), 'Background error')
    hdr['CONC'] = (float(results['concentration']), 'log10(r_tide / r_core)')
    hdr['CONC_ERR'] = (float(results['concentration_err']), 'Concentration error')
    # Carry every earlier-step parameter straight through, and add the
    # cluster center in real sky coordinates so no later script has to
    # convert arcmin offsets back to RA/Dec itself.
    if passthrough:
        for key, value in passthrough.items():
            hdr[key] = value

        if 'RAO' in passthrough and 'DECO' in passthrough:
            RAo = float(passthrough['RAO'])
            DECo = float(passthrough['DECO'])
            cos_DECo = float(passthrough.get(
                'COSDECO', np.cos(np.deg2rad(DECo))))
            hdr['CENT_RA'] = (RAo + x0 / (cos_DECo * 60.0),
                               'Cluster center RA, deg')
            hdr['CENT_DEC'] = (DECo + y0 / 60.0,
                                'Cluster center Dec, deg')

    hdr['ORIGIN'] = ('king_profile_104.py', 'Producing script')

    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdul = fits.HDUList([primary_hdu, table_hdu])

    fits_path = os.path.join(out_dir, f'king_fit_results_cluster_{cluster_id}.fits')
    hdul.writeto(fits_path, overwrite=True)
    return fits_path


# ============================================================
# 4. Print results (same style as the original script)
# ============================================================
def print_results(cluster_id, results):
    print(f"\nKing + background fit -- cluster {cluster_id}")
    print("-----------------------------")
    print(f"Amplitude    = {results['amplitude']:.3f} +/- "
          f"{results['amplitude_err']:.3f}")
    print(f"Core radius  = {results['rc']:.3f} +/- {results['rc_err']:.3f}")
    print(f"Tidal radius = {results['rt']:.3f} +/- {results['rt_err']:.3f}")
    print(f"Background   = {results['background']:.3f} +/- "
          f"{results['background_err']:.3f}")
    print(f"Concentration = {results['concentration']:.3f} +/- "
          f"{results['concentration_err']:.3f}")


# ============================================================
# 5. Plot (same style as the original script)
# ============================================================
def plot_results(cluster_id, results, r_max, out_dir):
    r_plot = np.linspace(0, r_max, 500)
    fit_model = results['fit_model']

    king_component = fit_model[0](r_plot)
    background_component = fit_model[1](r_plot)
    total_model = fit_model(r_plot)

    plt.figure(figsize=(8, 6))

    plt.errorbar(
        results['r_fit'], results['density_fit'],
        yerr=results['density_err_fit'],
        fmt="o", markersize=5, label="Stellar density"
    )
    plt.plot(r_plot, total_model, linewidth=2, label="King + background")
    plt.plot(r_plot, king_component, linestyle="--", label="King component")
    plt.plot(r_plot, background_component, linestyle=":", label="Background")
    plt.axvline(results['rc'], linestyle="--", label=r"$r_c$")
    plt.axvline(results['rt'], linestyle="-.", label=r"$r_t$")

    plt.xlabel("Radius, arcmin")
    plt.ylabel("Surface density [stars / arcmin$^2$]")
    plt.title(f"Cluster {cluster_id}: King + background fit")
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(out_dir, f'king_fit_cluster_{cluster_id}.png')
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    return png_path


# ============================================================
# Main interactive driver
# ============================================================
KING_SUMMARY_FIELDS = [
    'cluster_id', 'n_dbscan_members', 'n_stars_in_region',
    'dbscan_radius', 'centx', 'centy', 'cent_ra', 'cent_dec',
    'fit_a', 'fit_a_err', 'fit_b', 'fit_b_err',
    'angle', 'angle_err', 'ellipticity', 'ellipticity_err',
    'signal_fraction', 'signal_fraction_err',
    'king_rmax', 'king_nbins',
    'amplitude', 'amplitude_err', 'rcore', 'rcore_err',
    'rtide', 'rtide_err', 'background', 'background_err',
    'concentration', 'concentration_err',
]


def save_combined_king_outputs(rows, tile_meta, out_dir):
    """Writes king_all_clusters.fits and king_all_clusters.csv -- one row
    per cluster fitted in this run, carrying results from the
    clusterfinder, the ellipticity step and the King step together."""
    if not rows:
        return None, None

    int_fields = {'cluster_id', 'n_dbscan_members', 'n_stars_in_region',
                  'king_nbins'}
    cols = []
    for field in KING_SUMMARY_FIELDS:
        values = [r.get(field, np.nan) for r in rows]
        if field in int_fields:
            safe = [int(v) if np.isfinite(v) else -1 for v in values]
            cols.append(fits.Column(name=field, format='K',
                                     array=np.array(safe, dtype=np.int64)))
        else:
            cols.append(fits.Column(name=field, format='D',
                                     array=np.array(values, dtype=float)))
    table_hdu = fits.BinTableHDU.from_columns(cols, name='KING_ALL')

    hdr = fits.Header()
    hdr['NCLUST'] = (len(rows), 'Clusters in this combined file')
    if tile_meta:
        for key, value in tile_meta.items():
            hdr[key] = value
    hdr['ORIGIN'] = ('king_profile_104.py', 'Producing script')

    fits_path = os.path.join(out_dir, 'king_all_clusters.fits')
    fits.HDUList([fits.PrimaryHDU(header=hdr), table_hdu]).writeto(
        fits_path, overwrite=True)

    csv_path = os.path.join(out_dir, 'king_all_clusters.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=KING_SUMMARY_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in KING_SUMMARY_FIELDS})

    return fits_path, csv_path


# ============================================================
# Main interactive driver -- now works off the COMBINED ellipticity
# file for a whole tile, looping over every cluster in it.
# ============================================================
def run_interactive():
    print("king_profile_104.py -- King + background profile fit, for every "
          "cluster in a tile\n")

    combined_path = input(
        "Enter the path to the COMBINED ellipticity FITS for this tile "
        "(ellipse_all_clusters.fits): "
    ).strip().strip('"').strip("'")
    combined_path = os.path.expanduser(combined_path)
    if not os.path.isfile(combined_path):
        print(f"File not found: {combined_path}")
        sys.exit(1)

    with fits.open(combined_path) as hdul:
        chdr = hdul[0].header
        summary = hdul[1].data
        summary_names = summary.columns.names
        if 'cluster_id' not in summary_names:
            print(f"This file has no 'cluster_id' column -- is it really "
                  f"'ellipse_all_clusters.fits'? Columns: {summary_names}")
            sys.exit(1)

        tile_meta = {}
        for key in ('RAO', 'DECO', 'COSDECO'):
            if key in chdr:
                tile_meta[key] = chdr[key]

        rows_in = [{name: summary[name][i] for name in summary_names}
                   for i in range(len(summary))]

    out_dir = os.path.dirname(os.path.abspath(combined_path))
    print(f"Found {len(rows_in)} clusters in the combined file: "
          f"{[int(r['cluster_id']) for r in rows_in]}")

    n_bins_input = input(
        "\nNumber of radial bins, used for every cluster [default 24]: "
    ).strip()
    n_bins = int(n_bins_input) if n_bins_input else 24

    summary_rows = []
    processed = []

    for row_in in rows_in:
        cluster_id = int(row_in['cluster_id'])

        proceed = input(
            f"\nRun the King fit for cluster {cluster_id} "
            f"({int(row_in.get('n_dbscan_members', -1))} DBSCAN members)? "
            f"[Y/n]: "
        ).strip().lower()
        if proceed.startswith('n'):
            print(f"  Skipped cluster {cluster_id}.")
            continue

        # The per-cluster file holds the actual star positions; the
        # combined file only holds the fitted summary values.
        per_cluster = os.path.join(
            out_dir, f'ellipse_data_cluster_{cluster_id}.fits')
        if not os.path.isfile(per_cluster):
            print(f"  Cannot find {per_cluster} -- the King fit needs the "
                  f"per-cluster star positions, not just the summary row. "
                  f"Skipping cluster {cluster_id}.")
            continue

        (x, y, x0, y0, cid, half_width, cluster_label,
         passthrough) = load_cluster_fits(per_cluster)

        r_max = float(half_width) if half_width is not None else \
            float(np.percentile(np.sqrt((x - x0) ** 2 + (y - y0) ** 2), 95))

        try:
            results = fit_king_profile(x, y, x0, y0, r_max, n_bins)
        except RuntimeError as e:
            print(f"  Fit failed for cluster {cluster_id}: {e}")
            continue

        print_results(cluster_id, results)

        png_path = plot_results(cluster_id, results, r_max, out_dir)
        print(f"  Saved: {png_path}")

        fits_path = save_king_fits(cluster_id, results, x0, y0, r_max,
                                    n_bins, out_dir,
                                    passthrough=passthrough)
        print(f"  Saved: {fits_path}")

        def g(key, default=np.nan):
            v = row_in.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        summary_rows.append({
            'cluster_id': cluster_id,
            'n_dbscan_members': g('n_dbscan_members'),
            'n_stars_in_region': g('n_stars_in_region'),
            'dbscan_radius': g('dbscan_radius'),
            'centx': x0, 'centy': y0,
            'cent_ra': g('cent_ra'), 'cent_dec': g('cent_dec'),
            'fit_a': g('fit_a'), 'fit_a_err': g('fit_a_err'),
            'fit_b': g('fit_b'), 'fit_b_err': g('fit_b_err'),
            'angle': g('angle'), 'angle_err': g('angle_err'),
            'ellipticity': g('ellipticity'),
            'ellipticity_err': g('ellipticity_err'),
            'signal_fraction': g('signal_fraction'),
            'signal_fraction_err': g('signal_fraction_err'),
            'king_rmax': r_max, 'king_nbins': n_bins,
            'amplitude': results['amplitude'],
            'amplitude_err': results['amplitude_err'],
            'rcore': results['rc'], 'rcore_err': results['rc_err'],
            'rtide': results['rt'], 'rtide_err': results['rt_err'],
            'background': results['background'],
            'background_err': results['background_err'],
            'concentration': results['concentration'],
            'concentration_err': results['concentration_err'],
        })
        processed.append(cluster_id)

    combined_fits, combined_csv = save_combined_king_outputs(
        summary_rows, tile_meta, out_dir)

    print(f"\nDone. Processed clusters: {processed}")
    if combined_fits:
        print(f"\nCombined (whole-tile) outputs:")
        print(f"  {combined_fits}")
        print(f"  {combined_csv}")
        print("\nFeed the combined FITS into gmag_check_105.py to run the "
              "Gaia bright-star check on every cluster in this tile.")


if __name__ == "__main__":
    run_interactive()
