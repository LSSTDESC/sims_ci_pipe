"""
Module to compute visit-level PSF whisker plots.
"""
import itertools
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import lsst.geom as lsst_geom
from .ellipticity_distributions import get_point_sources


__all__ = ['get_e_components', 'get_sky_coords', 'psf_whisker_plot']


def get_e_components(ixx, iyy, ixy):
    """
    Compute ellipticity components from second moments.
    """
    e1 = (ixx - iyy)/(ixx + iyy)
    e2 = 2*ixy/(ixx + iyy)
    return e1, e2


def get_sky_coords(wcs, pixel_coords):
    """
    Compute the sky coordinates corresponding to a set of pixel
    coordinates.
    """
    ras, decs = [], []
    for pixel_coord in pixel_coords:
        sky_coord = wcs.pixelToSky(pixel_coord)
        ras.append(sky_coord[0].asDegrees())
        decs.append(sky_coord[1].asDegrees())
    return np.array(ras), np.array(decs)


def get_calexp_psf_ellipticity_components(datarefs, pixel_coords):
    """
    Get psf ellipticity components for a set of sensor-visit datarefs
    and pixel coordinates from the  PSF model in the calexps.

    Parameters
    ----------
    datarefs: lsst.daf.persistence.butlerSubset.ButlerSubset
        Data refs to the calexps in the desired visit.
    pixel_coords: list
        A list of lsst.geom.Point2D objects that contain the pixel
        coordinates at which to compute the ellipticity components.

    Returns
    -------
    4 lists of ra, dec, e1, and e2 values for plotting using
     matplotlib.quiver
    """
    ra_grid, dec_grid, e1_grid, e2_grid = [], [], [], []
    for dataref in list(datarefs):
        calexp = dataref.get('calexp')
        wcs = calexp.getWcs()
        psf = calexp.getPsf()
        for pixel_coord in pixel_coords:
            psf_shape = psf.computeShape(pixel_coord)
            e1, e2 = get_e_components(psf_shape.getIxx(), psf_shape.getIyy(),
                                      psf_shape.getIxy())
            e1_grid.append(e1)
            e2_grid.append(e2)
            sky_coord = wcs.pixelToSky(pixel_coord)
            ra_grid.append(sky_coord[0].asDegrees())
            dec_grid.append(sky_coord[1].asDegrees())
    return ra_grid, dec_grid, e1_grid, e2_grid


def get_interpolated_psf_ellipticity_components(datarefs, pixel_coords):
    """
    Get psf ellipticity components for a set of sensor-visit datarefs
    and pixel coordinates by interpolating (using scipy.interpolate.griddata)
    between the calib_psf_used stars for each exposure.

    Parameters
    ----------
    datarefs: lsst.daf.persistence.butlerSubset.ButlerSubset
        Data refs to the calexps in the desired visit.
    pixel_coords: list
        A list of lsst.geom.Point2D objects that contain the pixel
        coordinates at which to compute the ellipticity components.

    Returns
    -------
    4 lists of ra, dec, e1, and e2 values for plotting using
     matplotlib.quiver
    """
    ras, decs, e1s, e2s = [], [], [], []
    ra_grid, dec_grid = [], []
    for dataref in datarefs:
        ra_ccd_grid, dec_ccd_grid = get_sky_coords(dataref.get('calexp_wcs'),
                                                   pixel_coords)
        stars = get_point_sources(dataref.get('src'),
                                  flags=('calib_psf_used',))
        ra = [record['coord_ra'].asDegrees() for record in stars]
        dec = [record['coord_dec'].asDegrees() for record in stars]
        ixx = np.array([record['base_SdssShape_xx'] for record in stars])
        iyy = np.array([record['base_SdssShape_yy'] for record in stars])
        ixy = np.array([record['base_SdssShape_xy'] for record in stars])
        e1, e2 = get_e_components(ixx, iyy, ixy)

        ras.extend(ra)
        decs.extend(dec)
        e1s.extend(e1)
        e2s.extend(e2)
        ra_grid.extend(ra_ccd_grid)
        dec_grid.extend(dec_ccd_grid)

    e1_grid = griddata((ras, decs), e1s, (ra_grid, dec_grid), method='linear')
    e2_grid = griddata((ras, decs), e2s, (ra_grid, dec_grid), method='linear')
    return ra_grid, dec_grid, e1_grid, e2_grid


def psf_whisker_plot(butler, visit, scale=3, xy_pixels=None, use_calexp=True,
                     figsize=(8, 8)):
    """
    Make a psf whisker plot for a specified visit using the
    PSFs in the calexps or by interpolating the values using the
    calib_psf_stars

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Data butler for the repository containing single frame processing
        src catalogs.
    visit: int
        Visit to plot.
    scale: float [3]
        Scale of plotted whiskers.
    xy_pixels: list [None]
         Pixels in x- and y-directions on each CCD at which to compute
         the ellipticites.  If None, then `[0, 1000, 2000, 3000]` will be
         used.
    use_calexp: bool [True]
         Flag to use the PSF model available from the calexps.  If False,
         then interpolate using the calib_psf_stars.
    figsize: (float, float) [(8, 8)]
        Figure size in inches.
    """
    if xy_pixels is None:
        xy_pixels = [0, 1000, 2000, 3000]
    pixel_coords = [lsst_geom.Point2D(*_) for _ in
                    itertools.product(xy_pixels, xy_pixels)]

    datarefs = butler.subset('calexp', visit=visit)
    band = list(datarefs)[0].get('calexp_md').getScalar('FILTER')

    if use_calexp:
        ra, dec, e1, e2 \
            = get_calexp_psf_ellipticity_components(datarefs, pixel_coords)
    else:
        ra, dec, e1, e2 \
            = get_interpolated_psf_ellipticity_components(datarefs,
                                                          pixel_coords)

    _, ax = plt.subplots(1, 1, figsize=figsize)
    plt.axis('equal')

    qplot = ax.quiver(ra, dec, e1, e2, scale_units='x',
                      angles='xy', scale=scale, headaxislength=0,
                      headlength=0, headwidth=0)
    ax.quiverkey(qplot, 0.7, 0.95, 0.03, r'$e = 0.03$', labelpos='E',
                 coordinates='axes')

    plt.annotate(s='Visit: %d, filter: %s' % (visit, band),
                 xy=(0.1, 0.95), xycoords='axes fraction')
    ax.set_xlabel("RA (degrees)")
    ax.set_ylabel("Dec (degrees)")
