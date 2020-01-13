"""
Compute visit-level distributions of psf_mag - calib_mag to check
for biases in photometry.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import lsst.daf.persistence as dp
from .ellipticity_distributions import get_point_sources


__all__ = ['get_psf_calib_mags', 'psf_mag_check']


def get_psf_calib_mags(butler, visit, sn_min=150):
    """
    Compute psf and calib magnitudes.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Butler pointing at the data repo with the calexps.
    visit: int
        Visit number to consider.
    sn_min: float [150]
        Mininum signal-to-noise cut on psfFlux/psfFluxErr.

    Returns
    -------
    pandas.DataFrame containing the psf_mag and calib_mag values.
    """
    datarefs = butler.subset('src', visit=visit)
    psf_mags = []
    calib_mags = []
    psf_fluxes = []
    psf_fluxErrs = []
    for dataref in list(datarefs):
        try:
            src = dataref.get('src')
        except:
            continue
        photoCalib = dataref.get('calexp_photoCalib')
        visit = dataref.dataId['visit']
        stars = get_point_sources(src)
        psf_mags.extend(photoCalib.instFluxToMagnitude(
            stars, 'base_PsfFlux').transpose()[0])
        calib_mags.extend(photoCalib.instFluxToMagnitude(
            stars, 'base_CircularApertureFlux_12_0').transpose()[0])
        psf_fluxes.extend(stars['base_PsfFlux_instFlux'])
        psf_fluxErrs.extend(stars['base_PsfFlux_instFluxErr'])
    psf_mags = np.array(psf_mags)
    calib_mags = np.array(calib_mags)
    psf_fluxes = np.array(psf_fluxes)
    psf_fluxErrs = np.array(psf_fluxErrs)
    psf_flux_sn = psf_fluxes/psf_fluxErrs
    index = np.where((psf_flux_sn == psf_flux_sn) & (psf_flux_sn > sn_min))
    return pd.DataFrame(data=dict(psf_mag=psf_mags[index],
                                  calib_mag=calib_mags[index]))


def psf_mag_check(repo, visit, dmag_range=(-0.05, 0.05), sn_min=150):
    """
    Plot distribution of  delta_mag = psf_mag - calib_mag values, and
    return estimate of the delta_mag peak location.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Butler pointing at the data repo with the calexps.
    visit: int
        Visit number to consider.
    dmag_range: (float, float) [(-0.05, 0.05)]
        Magnitude range to use for plotting and median estimation.
    sn_min: float [150]
        Mininum signal-to-noise cut on psfFlux/psfFluxErr.

    Returns
    -------
    float: An estimate of the delta_mag peak location.
    """
    butler = dp.Butler(repo)
    df = get_psf_calib_mags(butler, visit, sn_min=sn_min)
    if len(df) == 0:
        return None
    delta_mag = (df['psf_mag'] - df['calib_mag']).to_numpy()
    delta_mag = delta_mag[np.where(delta_mag == delta_mag)]
    index = np.where((dmag_range[0] < delta_mag) & (delta_mag < dmag_range[1]))
    dmag_median = np.median(delta_mag[index])
    plt.hist(delta_mag, range=dmag_range, bins=100, histtype='step')
    plt.axvline(0, linestyle=':')
    plt.axvline(dmag_median, linestyle='--')
    plt.annotate(f'median: {dmag_median*1000:.2f} mmag\n'
                 f'psfFlux/psfFluxErr > {sn_min}', (0.05, 0.95),
                 xycoords='axes fraction', verticalalignment='top')
    plt.xlabel('psf_mag - calib_mag')

    return dmag_median
