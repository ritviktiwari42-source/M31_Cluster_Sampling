"""
gmag_check_105.py

Step (4) of the cluster-vetting chain: check whether a bright star sits
close to the cluster, which would suggest the "cluster" is really an
artifact in the diffraction spikes / halo of that bright star.

Uses the Gaia DR3 catalog (Gaia_DR3_tmp.fits, downloaded from VizieR
for G <= 19 mag) and reports the BRIGHTEST Gaia G magnitude found
within 1x, 5x and 10x the cluster size. The factors of 5 and 10 are
exploratory, per Dr. Ivanov: the hope is that a bright star showing up
within a few cluster radii flags a spurious detection.

SIMPLIFIED INPUTS:
Earlier this script also needed the DBSCAN export, purely to convert
the cluster's arcmin offsets back into RA/Dec. The clusterfinder now
stores the tile reference point (RAO/DECO) in its FITS header, the
ellipticity step copies it forward, and the King step uses it to write
CENT_RA / CENT_DEC directly. So this script now needs only TWO files:

  1. The King FITS for ONE cluster (king_fit_results_cluster_<id>.fits)
  2. The Gaia DR3 catalog (Gaia_DR3_tmp.fits)

Everything needed for the summary -- clusterfinder counts, ellipse fit
parameters and errors, King fit parameters -- already travels inside
the King file's header.

OUTPUTS (both per cluster, written next to the King input file):
  - gmag_check_cluster_<id>.fits : header with every parameter gathered
    from all four steps, plus a table of the Gaia stars inside 10x.
  - cluster_summary_<id>.csv : one-row summary combining results from
    (1) the clusterfinder, (2) the ellipticity fit, (3) the King fit,
    and (4) this G magnitude check.

USAGE:
    python gmag_check_105.py
"""

import os
import sys
import csv
import numpy as np
from astropy.io import fits


def ask_path(prompt, required=True):
    p = input(prompt).strip().strip('"').strip("'")
    if not p:
        if required:
            print("A path is required here.")
            sys.exit(1)
        return None
    p = os.path.expanduser(p)
    if not os.path.isfile(p):
        if required:
            print(f"File not found: {p}")
            sys.exit(1)
        print(f"  File not found: {p} -- continuing without it.")
        return None
    return p


def hget(hdr, key, default=np.nan):
    return hdr[key] if key in hdr else default


# ============================================================
# 1. Combined King FITS for a whole tile
# ============================================================
def load_combined_king(path):
    with fits.open(path) as hdul:
        chdr = hdul[0].header
        table = hdul[1].data
        names = table.columns.names
        if 'cluster_id' not in names:
            print(f"This file has no 'cluster_id' column -- is it really "
                  f"'king_all_clusters.fits'? Columns: {names}")
            sys.exit(1)
        rows = [{n: table[n][i] for n in names} for i in range(len(table))]
        tile_meta = {k: chdr[k] for k in ('RAO', 'DECO', 'COSDECO')
                     if k in chdr}
    return rows, tile_meta


# ============================================================
# 2. Gaia DR3
# ============================================================
def load_gaia(path):
    with fits.open(path) as hdul:
        data = hdul[1].data
        names = data.columns.names

        ra_key = 'RA_ICRS' if 'RA_ICRS' in names else '_RAJ2000'
        dec_key = 'DE_ICRS' if 'DE_ICRS' in names else '_DEJ2000'
        if 'Gmag' not in names:
            print(f"Gaia file has no 'Gmag' column. Available: {names}")
            sys.exit(1)

        ra = np.array(data[ra_key], dtype=float)
        dec = np.array(data[dec_key], dtype=float)
        gmag = np.array(data['Gmag'], dtype=float)
        source = (np.array(data['Source']) if 'Source' in names
                  else np.zeros(len(ra), dtype=np.int64))

    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(gmag)
    return ra[finite], dec[finite], gmag[finite], source[finite]


def angular_separation_arcmin(ra1_deg, dec1_deg, ra2, dec2):
    """Small-angle separation in arcmin, with the cos(dec) term applied
    to the RA difference. All searches here are far below a degree, so
    the flat-sky approximation is well within the precision needed."""
    cos_dec = np.cos(np.deg2rad(dec1_deg))
    dra = (ra2 - ra1_deg) * cos_dec
    ddec = (dec2 - dec1_deg)
    return np.sqrt(dra ** 2 + ddec ** 2) * 60.0


def brightest_within(sep_arcmin, gmag, source, radius):
    inside = sep_arcmin <= radius
    n = int(inside.sum())
    if n == 0:
        return {'radius_arcmin': radius, 'n_gaia': 0,
                'brightest_gmag': np.nan, 'brightest_source': 0,
                'brightest_sep_arcmin': np.nan}
    idx_local = np.argmin(gmag[inside])
    return {
        'radius_arcmin': radius,
        'n_gaia': n,
        'brightest_gmag': float(gmag[inside][idx_local]),
        'brightest_source': int(source[inside][idx_local]),
        'brightest_sep_arcmin': float(sep_arcmin[inside][idx_local]),
    }


# ============================================================
# Main
# ============================================================
SUMMARY_FIELDS = [
    'cluster_id',
    # (1) clusterfinder
    'cf_n_members', 'cf_n_stars_in_region', 'cf_radius_arcmin',
    'cf_center_x_arcmin', 'cf_center_y_arcmin',
    'cf_center_ra_deg', 'cf_center_dec_deg',
    # (2) ellipticity
    'ellipse_semi_a', 'ellipse_semi_a_err',
    'ellipse_semi_b', 'ellipse_semi_b_err',
    'ellipse_angle', 'ellipse_angle_err',
    'ellipse_ellipticity', 'ellipse_ellipticity_err',
    'ellipse_signal_fraction', 'ellipse_signal_fraction_err',
    # (3) King
    'king_rmax', 'king_nbins', 'king_amplitude', 'king_amplitude_err',
    'king_rcore', 'king_rcore_err', 'king_rtide', 'king_rtide_err',
    'king_background', 'king_background_err',
    'king_concentration', 'king_concentration_err',
    # (4) G magnitude
    'gmag_size_key', 'gmag_cluster_size_arcmin',
    'gmag_r1x_arcmin', 'gmag_n_gaia_1x', 'gmag_brightest_1x',
    'gmag_brightest_sep_1x', 'gmag_brightest_source_1x',
    'gmag_r5x_arcmin', 'gmag_n_gaia_5x', 'gmag_brightest_5x',
    'gmag_brightest_sep_5x', 'gmag_brightest_source_5x',
    'gmag_r10x_arcmin', 'gmag_n_gaia_10x', 'gmag_brightest_10x',
    'gmag_brightest_sep_10x', 'gmag_brightest_source_10x',
]


def save_individual_gmag_fits(cluster_id, row, results, cluster_ra,
                                cluster_dec, cluster_size, size_choice,
                                gaia_subset, out_dir):
    g_source, g_ra, g_dec, g_mag, sep = gaia_subset
    table_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='Source', format='K',
                    array=g_source.astype(np.int64)),
        fits.Column(name='RA_deg', format='D', array=g_ra),
        fits.Column(name='DEC_deg', format='D', array=g_dec),
        fits.Column(name='Gmag', format='D', array=g_mag),
        fits.Column(name='sep_arcmin', format='D', array=sep),
    ], name='GAIA_NEAR_CLUSTER')

    def safe(v, fallback=-999.0):
        # FITS headers cannot store NaN. A cluster with no Gaia star
        # inside a given radius legitimately has no "brightest" value,
        # so store a sentinel and let the count column (NGAIA*X = 0)
        # be the thing that tells you it was empty.
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return fallback
        return fv if np.isfinite(fv) else fallback

    hdr = fits.Header()
    hdr['CLUSTID'] = (int(cluster_id), 'DBSCAN cluster ID')
    hdr['CENT_RA'] = (float(cluster_ra), 'Cluster center RA, deg')
    hdr['CENT_DEC'] = (float(cluster_dec), 'Cluster center Dec, deg')
    hdr['SIZEKEY'] = (size_choice, 'Which radius was used as 1x')
    hdr['CLSIZE'] = (float(cluster_size), 'Cluster size (1x), arcmin')
    for factor in (1, 5, 10):
        r = results[factor]
        hdr[f'R{factor}X'] = (float(r['radius_arcmin']),
                               f'{factor}x search radius, arcmin')
        hdr[f'NGAIA{factor}X'] = (int(r['n_gaia']),
                                    f'Gaia stars within {factor}x')
        hdr[f'GMIN{factor}X'] = (safe(r['brightest_gmag']),
                                   f'Brightest Gmag within {factor}x '
                                   f'(-999 if none)')
        hdr[f'GSEP{factor}X'] = (safe(r['brightest_sep_arcmin']),
                                   f'Sep of brightest, arcmin '
                                   f'(-999 if none)')
    hdr['ORIGIN'] = ('gmag_check_105.py', 'Producing script')

    path = os.path.join(out_dir, f'gmag_check_cluster_{cluster_id}.fits')
    fits.HDUList([fits.PrimaryHDU(header=hdr), table_hdu]).writeto(
        path, overwrite=True)
    return path


def save_combined_outputs(rows, tile_meta, out_dir):
    """Writes gmag_all_clusters.fits and the final cluster_summary_all.csv
    -- one row per cluster, combining all four steps."""
    if not rows:
        return None, None

    int_fields = {'cluster_id', 'cf_n_members', 'cf_n_stars_in_region',
                  'king_nbins', 'gmag_n_gaia_1x', 'gmag_n_gaia_5x',
                  'gmag_n_gaia_10x', 'gmag_brightest_source_1x',
                  'gmag_brightest_source_5x', 'gmag_brightest_source_10x'}
    str_fields = {'gmag_size_key'}

    cols = []
    for field in SUMMARY_FIELDS:
        values = [r.get(field, np.nan) for r in rows]
        if field in str_fields:
            width = max(1, max(len(str(v)) for v in values))
            cols.append(fits.Column(name=field, format=f'{width}A',
                                     array=np.array([str(v) for v in values])))
        elif field in int_fields:
            safe = []
            for v in values:
                try:
                    safe.append(int(v) if np.isfinite(float(v)) else -1)
                except (TypeError, ValueError):
                    safe.append(-1)
            cols.append(fits.Column(name=field, format='K',
                                     array=np.array(safe, dtype=np.int64)))
        else:
            cols.append(fits.Column(name=field, format='D',
                                     array=np.array(values, dtype=float)))
    table_hdu = fits.BinTableHDU.from_columns(cols, name='CLUSTER_SUMMARY')

    hdr = fits.Header()
    hdr['NCLUST'] = (len(rows), 'Clusters in this combined file')
    if tile_meta:
        for key, value in tile_meta.items():
            hdr[key] = value
    hdr['ORIGIN'] = ('gmag_check_105.py', 'Producing script')

    fits_path = os.path.join(out_dir, 'gmag_all_clusters.fits')
    fits.HDUList([fits.PrimaryHDU(header=hdr), table_hdu]).writeto(
        fits_path, overwrite=True)

    csv_path = os.path.join(out_dir, 'cluster_summary_all.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in SUMMARY_FIELDS})

    return fits_path, csv_path


# ============================================================
# Main
# ============================================================
def run_interactive():
    print("gmag_check_105.py -- Gaia bright-star check for every cluster "
          "in a tile, plus the final combined summary\n")

    king_path = ask_path(
        "Enter the path to the COMBINED King FITS for this tile "
        "(king_all_clusters.fits): "
    )
    rows_in, tile_meta = load_combined_king(king_path)
    print(f"  Found {len(rows_in)} clusters: "
          f"{[int(r['cluster_id']) for r in rows_in]}")

    gaia_path = ask_path(
        "\nEnter the path to the Gaia DR3 catalog (Gaia_DR3_tmp.fits): "
    )
    g_ra, g_dec, g_mag, g_source = load_gaia(gaia_path)
    print(f"  Loaded {len(g_ra)} Gaia stars.")

    print("\nWhich radius should count as 'the cluster size' (1x)?")
    print("   dbscan_radius | king_rtide | king_rcore | ellipse_semi_a")
    size_choice = input("  [default dbscan_radius]: ").strip()
    if size_choice not in ('dbscan_radius', 'king_rtide', 'king_rcore',
                            'ellipse_semi_a'):
        size_choice = 'dbscan_radius'
    size_key_map = {
        'dbscan_radius': 'dbscan_radius',
        'king_rtide': 'rtide',
        'king_rcore': 'rcore',
        'ellipse_semi_a': 'fit_a',
    }
    size_col = size_key_map[size_choice]
    print(f"  Using {size_choice} as 1x.")

    out_dir = os.path.dirname(os.path.abspath(king_path))
    summary_rows = []

    for row_in in rows_in:
        cluster_id = int(row_in['cluster_id'])

        def g(key, default=np.nan):
            v = row_in.get(key, default)
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        cluster_ra = g('cent_ra')
        cluster_dec = g('cent_dec')
        if not (np.isfinite(cluster_ra) and np.isfinite(cluster_dec)):
            if tile_meta and 'RAO' in tile_meta and 'DECO' in tile_meta:
                RAo = float(tile_meta['RAO'])
                DECo = float(tile_meta['DECO'])
                cos_DECo = float(tile_meta.get(
                    'COSDECO', np.cos(np.deg2rad(DECo))))
                cluster_ra = RAo + g('centx') / (cos_DECo * 60.0)
                cluster_dec = DECo + g('centy') / 60.0
            else:
                print(f"\n  Cluster {cluster_id}: no sky coordinates "
                      f"available -- skipping.")
                continue

        cluster_size = g(size_col)
        if not np.isfinite(cluster_size) or cluster_size <= 0:
            print(f"\n  Cluster {cluster_id}: '{size_choice}' is not usable "
                  f"({cluster_size}) -- skipping.")
            continue

        sep = angular_separation_arcmin(cluster_ra, cluster_dec, g_ra, g_dec)

        results = {}
        for factor in (1, 5, 10):
            results[factor] = brightest_within(
                sep, g_mag, g_source, cluster_size * factor)

        r1 = results[1]
        print(f"\n  Cluster {cluster_id}: 1x={cluster_size:.4f} arcmin, "
              f"brightest G within 1x = "
              f"{r1['brightest_gmag'] if r1['n_gaia'] else 'none'}"
              + (f" at {r1['brightest_sep_arcmin']:.4f} arcmin"
                 if r1['n_gaia'] else ""))
        for factor in (5, 10):
            r = results[factor]
            print(f"      {factor:2d}x ({r['radius_arcmin']:.4f} arcmin): "
                  f"{r['n_gaia']} Gaia stars, brightest G="
                  + (f"{r['brightest_gmag']:.3f}" if r['n_gaia'] else "none"))

        inside10 = sep <= cluster_size * 10
        ind_path = save_individual_gmag_fits(
            cluster_id, row_in, results, cluster_ra, cluster_dec,
            cluster_size, size_choice,
            (g_source[inside10], g_ra[inside10], g_dec[inside10],
             g_mag[inside10], sep[inside10]),
            out_dir)
        print(f"      Saved: {ind_path}")

        row = {'cluster_id': cluster_id}
        row['cf_n_members'] = g('n_dbscan_members')
        row['cf_n_stars_in_region'] = g('n_stars_in_region')
        row['cf_radius_arcmin'] = g('dbscan_radius')
        row['cf_center_x_arcmin'] = g('centx')
        row['cf_center_y_arcmin'] = g('centy')
        row['cf_center_ra_deg'] = cluster_ra
        row['cf_center_dec_deg'] = cluster_dec
        row['ellipse_semi_a'] = g('fit_a')
        row['ellipse_semi_a_err'] = g('fit_a_err')
        row['ellipse_semi_b'] = g('fit_b')
        row['ellipse_semi_b_err'] = g('fit_b_err')
        row['ellipse_angle'] = g('angle')
        row['ellipse_angle_err'] = g('angle_err')
        row['ellipse_ellipticity'] = g('ellipticity')
        row['ellipse_ellipticity_err'] = g('ellipticity_err')
        row['ellipse_signal_fraction'] = g('signal_fraction')
        row['ellipse_signal_fraction_err'] = g('signal_fraction_err')
        row['king_rmax'] = g('king_rmax')
        row['king_nbins'] = g('king_nbins')
        row['king_amplitude'] = g('amplitude')
        row['king_amplitude_err'] = g('amplitude_err')
        row['king_rcore'] = g('rcore')
        row['king_rcore_err'] = g('rcore_err')
        row['king_rtide'] = g('rtide')
        row['king_rtide_err'] = g('rtide_err')
        row['king_background'] = g('background')
        row['king_background_err'] = g('background_err')
        row['king_concentration'] = g('concentration')
        row['king_concentration_err'] = g('concentration_err')
        row['gmag_size_key'] = size_choice
        row['gmag_cluster_size_arcmin'] = cluster_size
        for factor in (1, 5, 10):
            r = results[factor]
            row[f'gmag_r{factor}x_arcmin'] = r['radius_arcmin']
            row[f'gmag_n_gaia_{factor}x'] = r['n_gaia']
            row[f'gmag_brightest_{factor}x'] = r['brightest_gmag']
            row[f'gmag_brightest_sep_{factor}x'] = r['brightest_sep_arcmin']
            row[f'gmag_brightest_source_{factor}x'] = r['brightest_source']
        summary_rows.append(row)

    combined_fits, combined_csv = save_combined_outputs(
        summary_rows, tile_meta, out_dir)

    print(f"\nDone. {len(summary_rows)} clusters processed.")
    if combined_fits:
        print(f"\nFinal combined outputs:")
        print(f"  {combined_fits}")
        print(f"  {combined_csv}   <-- final summary of steps 1-4")


if __name__ == "__main__":
    run_interactive()
