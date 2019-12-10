"""
Module to produce PSF ellipticity plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import lsst.daf.persistence as dp
from .opsim_db_interface import OpSimDb


__all__ = ['get_point_sources', 'plot_ellipticities',
           'ellipticity_distributions']


def asymQ(ixx, iyy, ixy):
    asymQx = ixx - iyy
    asymQy = 2*ixy
    return np.sqrt(asymQx**2 + asymQy**2)


def trQ(ixx, iyy):
    return ixx + iyy


def get_a(ixx, iyy, ixy):
    return np.sqrt(0.5*(trQ(ixx, iyy) + asymQ(ixx, iyy, ixy)))


def get_b(ixx, iyy, ixy):
    return np.sqrt(0.5*(trQ(ixx, iyy) - asymQ(ixx, iyy, ixy)))


def get_e(ixx, iyy, ixy):
    a = get_a(ixx, iyy, ixy)
    b = get_b(ixx, iyy, ixy)
    return (a**2 - b**2)/(a**2 + b**2)


def get_point_sources(src, flux_type='base_PsfFlux', flags=()):
    ext = src.get('base_ClassificationExtendedness_value')
    model_flag = src.get(f'{flux_type}_flag')
    model_flux = src.get(f'{flux_type}_instFlux')
    num_children = src.get('deblend_nChild')
    condition = ((ext == 0) &
                 (model_flag == False) &
                 (model_flux > 0) &
                 (num_children == 0))
    for flag in flags:
        values = src.get(flag)
        condition &= (values == True)
    return src.subset(condition)


def plot_ellipticities(butler, visits, opsim_db_file=None, min_altitude=80.,
                       seeing_range=(0.65, 0.75), e_range=(0, 0.1), bins=100):
    """
    Plot the ellipticity distributions derived from the stars in the
    provided visits and compare to the LPM-17 median and 95 percentile
    design limits.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Data butler pointing at the repo with the visit data.
    visits: list-like
        List of visits to consider.
    opsim_db_file: str [None]
        Opsim db file to use to determine if a visit pointing has the
        required seeing and altitudes.
    min_altitude: float [80.]
        Minimum altitude constraint on visit pointing in degrees.
    seeing_range: (float, float) [(0.65, 0.75)]
        Range of acceptable seeing values (compared to FWHMtot from
        Document-20160).
    e_range: (float, float) [(0, 0.1)]
        Ellipticity range for plot.
    bins: int [100]
        Number of bins for ellipticity histogram.
    """
    opsim_db = OpSimDb(opsim_db_file)
    ellipticities = []
    for visit in visits:
        row = opsim_db(visit)
        if (np.degrees(row.altitude) < min_altitude or
            not (seeing_range[0] < row.FWHMgeom < seeing_range[1])):
            continue
        datarefs = butler.subset('src', visit=int(visit))
        for i, dataref in enumerate(datarefs):
            try:
                src = get_point_sources(dataref.get('src'))
            except dp.butlerExceptions.NoResults:
                continue
            for record in src:
                ellipticities.append(get_e(record['base_SdssShape_psf_xx'],
                                           record['base_SdssShape_psf_yy'],
                                           record['base_SdssShape_psf_xy']))

    plt.hist(ellipticities, range=e_range, bins=bins, histtype='step')
    e_median = np.median(ellipticities)
    e_95 = np.percentile(ellipticities, 95)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axvline(e_median, linestyle=':', color='red')
    plt.axvline(0.04, linestyle='--', color='red')
    plt.axvline(e_95, linestyle=':', color='green')
    plt.axvline(0.07, linestyle='--', color='green')
    plt.xlabel(r'$|e| = (1 - q^{2})/(1 + q^{2})$')


def ellipticity_distributions(repo, outfile=None, opsim_db=None):
    """Plot the ellipticity distribuions for r- and i-band."""
    butler = dp.Butler(repo)
    fig = plt.figure(figsize=(5, 8))
    for i, band in enumerate(('r', 'i')):
        fig.add_subplot(2, 1, i+1)
        visits = set([_.dataId['visit'] for _ in
                      butler.subset('src', filter=band)])
        plot_ellipticities(butler, visits, opsim_db_file=opsim_db)
        plt.title(f'Run2.2i, {band}-band, {len(visits)} visits')
    plt.tight_layout()
    if outfile is None:
        outfile = f'ellipticity_distributions.png'
    plt.savefig(outfile)
