"""
Compute visit-level distributions of psf_mag - calib_mag to check
for biases in photometry.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.mixture
import lsst.daf.persistence as dp
from .ellipticity_distributions import get_point_sources


__all__ = ['get_psf_calib_mags', 'psf_mag_check']


def get_psf_calib_mags(butler, visit):
    """
    Compute psf and calib magnitudes.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Butler pointing at the data repo with the calexps.
    visit: int
        Visit number to consider.

    Returns
    -------
    pandas.DataFrame containing the psf_mag and calib_mag values.
    """
    datarefs = butler.subset('src', visit=visit)
    psf_mags = []
    calib_mags = []
    for dataref in list(datarefs):
        try:
            src = dataref.get('src')
        except:
            break
        photoCalib = dataref.get('calexp_photoCalib')
        visit = dataref.dataId['visit']
        stars = get_point_sources(src)
        psf_mags.extend(photoCalib.instFluxToMagnitude(
            stars, 'base_PsfFlux').transpose()[0])
        calib_mags.extend(photoCalib.instFluxToMagnitude(
            stars, 'base_CircularApertureFlux_12_0').transpose()[0])
    return pd.DataFrame(data=dict(psf_mag=psf_mags, calib_mag=calib_mags))


def fit_gmm(x, n_components=2):
    """
    Fit a multicomponent Gaussian mixture model to the distribution
    and return the fit results for each component, ordered by weight,
    descending.

    Parameters
    ----------
    x: list-like
        The values for which the distribution will be modeled.
    n_components: int [2]
        Number of Gaussian components to include in the model.

    Returns
    -------
    n_components tuple of (weight, mean, sigma) for each component
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=n_components,
                                          warm_start=True)
    gmm = gmm.fit(x[:, np.newaxis])
    gmm = gmm.fit(x[:, np.newaxis])
    gmm = gmm.fit(x[:, np.newaxis])
    component_pars = []
    for weight, mean, covar in zip(gmm.weights_.ravel(), gmm.means_.ravel(),
                                   gmm.covariances_.ravel()):
        component_pars.append((weight, mean, np.sqrt(covar)))
    return sorted(component_pars, key=lambda x: x[0], reverse=True)


def psf_mag_check(repo, visit, outfile=None, figsize=(6, 4)):
    """
    Plot distribution of  delta_mag = psf_mag - calib_mag values, and
    return estimate of the delta_mag peak location.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Butler pointing at the data repo with the calexps.
    visit: int
        Visit number to consider.
    outfile: str [None]
        Output filename for plot. If None, then 'delta_mag_{visit

    Returns
    -------
    float: An estimate of the delta_mag peak location.
    """
    butler = dp.Butler(repo)
    fig = plt.figure(figsize=figsize)
    df = get_psf_calib_mags(butler, visit)
    if len(df) == 0:
        return None
    delta_mag = df['psf_mag'] - df['calib_mag']
    _, dmag_peak, _ = fit_gmm(delta_mag, n_components=2)[0]
    plt.hist(delta_mag, range=(-0.05, 0.05), bins=100, histtype='step')
    plt.axvline(0, linestyle=':')
    plt.axvline(dmag_peak, linestyle='--')
    plt.annotate(f'peak: {dmag_peak*1000:.2f} mmag', (0.1, 0.95),
                 xycoords='axes fraction')
    plt.xlabel('psf_mag - calib_mag')
    plt.legend(fontsize='x-small')

    if outfile is None:
        band = list(butler.subset('src', visit=visit))[0].dataId['filter']
        outfile = f'delta_mag_v{visit}-{band}.png'
    plt.savefig(outfile)

    return dmag_peak
