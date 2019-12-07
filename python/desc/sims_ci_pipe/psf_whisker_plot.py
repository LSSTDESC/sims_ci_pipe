import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import lsst.daf.persistence as dp
from .ellipticity_distributions import get_point_sources


__all__ = ['get_e_components', 'psf_whisker_plot']


def get_e_components(ixx, iyy, ixy):
    """
    Compute ellipticity components from second moments of PSF.
    """
    denominator = ixx**2 + iyy**2
    e1 = (ixx**2 - iyy**2)/denominator
    e2 = 2*ixy**2/denominator
    return e1, e2


def psf_whisker_plot(repo, visit, grid_shape=(50, 50), figsize=(8, 8)):
    """
    Make a psf whisker plot for a specified visit.

    Parameters
    ----------
    repo: str
        Data repository containing single frame processing src catalogs.
    visit: int
        Visit to plot.
    grid_shape: (int, int) [(50, 50)]
        Numbers of bins in x and y over which to average e1 and e2 values.
    figsize: (float, float) [(8, 8)]
        Figure size in inches.
    """
    butler = dp.Butler(repo)
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
        ixx = np.array([record['base_SdssShape_psf_xx'] for record in stars])
        iyy = np.array([record['base_SdssShape_psf_yy'] for record in stars])
        ixy = np.array([record['base_SdssShape_psf_xy'] for record in stars])
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
    ipts = np.vstack(a.ravel() for a in np.meshgrid(yi, xi)[::-1]).T

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
                  angles='xy', scale=0.3,
                  headaxislength=0, headlength=0, headwidth=0)

    ax.quiverkey(Q, 0.85, 0.85, 0.03, r'$e = 0.03$', labelpos='E',
                 coordinates='figure', fontproperties={'size':14})

    xmean = np.mean(ra)
    ymean = np.mean(dec)
#    ax.set_xlim(xmean - 3, xmean + 3)
#    ax.set_ylim(ymean - 3, ymean + 3)
    plt.annotate(s='Visit: %d, filter: %s' % (visit, band),
                 xy=(xmean - 2.0, ymean + 2.5))
    ax.set_xlabel("RA [deg.]", fontsize=16)
    ax.set_ylabel("Dec [deg.]", fontsize=16)
