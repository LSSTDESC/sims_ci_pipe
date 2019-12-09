import itertools
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import lsst.geom as lsst_geom
from .ellipticity_distributions import get_point_sources


__all__ = ['get_e_components', 'calexp_psf_whisker_plot', 'psf_whisker_plot']


def get_e_components(ixx, iyy, ixy):
    """
    Compute ellipticity components from second moments of PSF.
    """
    denominator = ixx**2 + iyy**2
    e1 = (ixx**2 - iyy**2)/denominator
    e2 = 2*ixy**2/denominator
    return e1, e2


def calexp_psf_whisker_plot(butler, visit, figsize=(8, 8), scale=3,
                            xy_pixels=None):
    """
    Make a psf whisker plot for a specified visit using the
    PSFs fitted by the LSST Stack.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Data butler for the repository containing single frame processing
        src catalogs.
    visit: int
        Visit to plot.
    figsize: (float, float) [(8, 8)]
        Figure size in inches.
    scale: float [3]
        Scale of plotted whiskers.
    xy_pixels: list [None]
         Pixels in x- and y-directions on each CCD at which to compute
         the ellipticites.  If None, then `[0, 1000, 2000, 3000]` will be
         used.
    """
    if xy_pixels is None:
        xy_pixels = range(0, 4000, 1000)
    pixel_coords = [lsst_geom.Point2D(*_) for _ in
                    itertools.product(xy_pixels, xy_pixels)]
    ras, decs, e1s, e2s = [], [], [], []
    datarefs = butler.subset('calexp', visit=visit)
    band = None
    for dataref in list(datarefs):
        if band is None:
            md = dataref.get('calexp_md')
            band = md.getScalar('FILTER')
        calexp = dataref.get('calexp')
        wcs = calexp.getWcs()
        psf = calexp.getPsf()
        for pixel_coord in pixel_coords:
            psf_shape = psf.computeShape(pixel_coord)
            e1, e2 = get_e_components(psf_shape.getIxx(), psf_shape.getIyy(),
                                      psf_shape.getIxy())
            e1s.append(e1)
            e2s.append(e2)
            sky_coord = wcs.pixelToSky(pixel_coord)
            ras.append(sky_coord[0].asDegrees())
            decs.append(sky_coord[1].asDegrees())

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    qplot = ax.quiver(ras, decs, e1s, e2s, scale_units='x', angles='xy',
                      scale=scale, headaxislength=0, headlength=0, headwidth=0)
    ax.quiverkey(qplot, 0.85, 0.85, 0.03, r'$e = 0.03$', labelpos='E',
                 coordinates='figure', fontproperties={'size': 14})

    xmean = np.mean(ras)
    ymean = np.mean(decs)
    plt.annotate(s='Visit: %d, filter: %s' % (visit, band),
                 xy=(xmean - 2.0, ymean + 2.5))
    ax.set_xlabel("RA [deg.]", fontsize=16)
    ax.set_ylabel("Dec [deg.]", fontsize=16)


def psf_whisker_plot(butler, visit, scale=3, grid_shape=(50, 50),
                     figsize=(8, 8)):
    """
    Make a psf whisker plot for a specified visit.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Data butler for the repository containing single frame processing
        src catalogs.
    visit: int
        Visit to plot.
    scale: float [3]
        Scale of plotted whiskers.
    grid_shape: (int, int) [(50, 50)]
        Numbers of bins in x and y over which to average e1 and e2 values.
    figsize: (float, float) [(8, 8)]
        Figure size in inches.
    """
    datarefs = butler.subset('src', visit=visit)
    band = None

    ras, decs, e1s, e2s = [], [], [], []
    for dataref in datarefs:
        if band is None:
            md = dataref.get('calexp_md')
            band = md.getScalar('FILTER')
        stars = get_point_sources(dataref.get('src'))
        ras.append([record['coord_ra'].asDegrees() for record in stars])
        decs.append([record['coord_dec'].asDegrees() for record in stars])
        ixx = np.array([record['base_SdssShape_xx'] for record in stars])
        iyy = np.array([record['base_SdssShape_yy'] for record in stars])
        ixy = np.array([record['base_SdssShape_xy'] for record in stars])
        e1, e2 = get_e_components(ixx, iyy, ixy)
        e1s.append(e1)
        e2s.append(e2)

    ra = np.concatenate(ras)
    dec = np.concatenate(decs)
    e1 = np.concatenate(e1s)
    e2 = np.concatenate(e2s)

    nx, ny = grid_shape

    # (N, 2) arrays of input x, y coords and u, v values
    pts = np.vstack((ra, dec)).T
    vals = np.vstack((e1, e2)).T

    # The new x and y coordinates for the grid, which will correspond
    # to the columns and rows of u and v respectively
    xi = np.linspace(ra.min(), ra.max(), nx)
    yi = np.linspace(dec.min(), dec.max(), ny)

    # an (nx*ny, 2) array of x, y coordinates to interpolate at
    ipts = np.vstack([a.ravel() for a in np.meshgrid(yi, xi)[::-1]]).T

    # an (nx*ny, 2) array of interpolated u, v values
    ivals = griddata(pts, vals, ipts, method='linear')

    # reshape interpolated u, v values into (ny, nx) arrays
    ui, vi = ivals.T
    ui.shape = vi.shape = (ny, nx)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.axis('equal')

    mask = np.ones(len(ipts[:,0]), dtype=bool)

    Q = ax.quiver(ipts[:,0][mask], ipts[:,1][mask],
                  ui.flatten()[mask], vi.flatten()[mask], scale_units='x',
                  angles='xy',
                  scale=scale,
                  headaxislength=0, headlength=0, headwidth=0)

    ax.quiverkey(Q, 0.85, 0.85, 0.03, r'$e = 0.03$', labelpos='E',
                 coordinates='figure', fontproperties={'size':14})

    xmean = np.mean(ra)
    ymean = np.mean(dec)
    plt.annotate(s='Visit: %d, filter: %s' % (visit, band),
                 xy=(xmean - 2.0, ymean + 2.5))
    ax.set_xlabel("RA [deg.]", fontsize=16)
    ax.set_ylabel("Dec [deg.]", fontsize=16)
