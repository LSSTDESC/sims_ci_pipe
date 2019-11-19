"""
Module to create single frame processing validation plots.
"""
import os
from collections import defaultdict
import numpy as np
from scipy.stats import binned_statistic
import pandas as pd
import healpy as hp
import lsst.afw.table as afw_table
import lsst.geom as lsst_geom
import lsst.daf.persistence as dp
from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask


class RefCat:
    """
    Class to provide access to sky cone selected reference catalog data
    for DC2.
    """
    def __init__(self, butler):
        """
        Parameters
        ----------
        butler: lsst.daf.persistence.Butler
            Butler pointing to the data repo with the processed visits.
        """
        self.butler = butler
        refConfig = LoadIndexedReferenceObjectsTask.ConfigClass()
        refConfig.filterMap = {_: f'lsst_{_}_smeared' for _ in 'ugrizy'}
        self.ref_task = LoadIndexedReferenceObjectsTask(self.butler,
                                                        config=refConfig)
    def ccd_center(self, dataId):
        """
        Return the CCD center for the selected sensor.

        Parameters
        ----------
        dataId: dict
            Complete dataId pointing to a specific sensor-visit.

        Returns
        -------
        lsst.geom.SpherePoint
            Sky coordinates corresponding to the center of the CCD.
        """
        calexp = self.butler.get('calexp', dataId)
        wcs = calexp.getWcs()
        dim = calexp.getDimensions()
        centerPixel = lsst_geom.Point2D(dim.getX()/2., dim.getY()/2.)
        return wcs.pixelToSky(centerPixel)

    def __call__(self, centerCoord, band, radius=2.1):
        """
        Return the reference catalog entries within the desired
        sky cone.

        Paramters
        ---------
        centerCoord: lsst.geom.SpherePoint
            Center of the sky cone to use to select objects.
        band: str
            Band of visit, one of 'ugrizy'.
        radius: float [2.1]
            Radius in degrees of the sky cone.

        Returns
        -------
        lsst.afw.table.SimpleCatalog
        """
        radius = lsst_geom.Angle(radius, lsst_geom.degrees)
        return self.ref_task.loadSkyCircle(centerCoord, radius, band).refCat


def point_source_matches(dataref, ref_cat0, max_offset=0.1,
                         src_columns=(), ref_columns=(),
                         flux_type='base_PsfFlux'):
    """
    Match point sources between a reference catalog and the dataref
    pointing to a src catalog.

    Parameters
    ----------
    dataref: lsst.daf.persistence.butlerSubset.ButlerDataref
        Dataref pointing to the desired sensor-visit.
    ref_cat0: lsst.afw.table.SimpleCatalog
        The reference catalog.
    max_offset: float [0.1]
        Maximum offset for positional matching in arcseconds.
    src_columns: list-like [()]
        Columns from the src catalog to save in the output dataframe.
    ref_columns: list-like [()]
        Columns from the reference catalog to save in the output dataframe.
        The column names will have 'ref_' prepended.
    flux_type: str ['base_PsfFlux']
        Flux type for point sources.

    Returns
    -------
    pandas.DataFrame
    """
    flux_col = f'{flux_type}_instFlux'
    src0 = dataref.get('src')
    band = dataref.dataId['filter']

    # Apply point source selections to the source catalog.
    ext = src0.get('base_ClassificationExtendedness_value')
    model_flag = src0.get(f'{flux_type}_flag')
    model_flux = src0.get(flux_col)
    num_children = src0.get('deblend_nChild')
    src = src0.subset((ext == 0) &
                      (not model_flag) &
                      (model_flux > 0) &
                      (num_children == 0))

    # Match RA, Dec with the reference catalog stars.
    ref_cat = ref_cat0.subset((ref_cat0.get('resolved') == 0))
    radius = lsst_geom.Angle(max_offset, lsst_geom.arcseconds)
    matches = afw_table.matchRaDec(ref_cat, src, radius)
    num_matches = len(matches)

    offsets = np.zeros(num_matches, dtype=np.float)
    ref_mags = np.zeros(num_matches, dtype=np.float)
    src_mags = np.zeros(num_matches, dtype=np.float)
    ref_data = defaultdict(list)
    src_data = defaultdict(list)
    calib = dataref.get('calexp_photoCalib')
    for i, match in enumerate(matches):
        offsets[i] = np.degrees(match.distance)*3600*1000.
        ref_mags[i] = match.first[f'lsst_{band}']
        src_mags[i] = calib.instFluxToMagnitude(match.second[flux_col])
        for ref_col in ref_columns:
            ref_data[f'ref_{ref_col}'].append(match.first[ref_col])
        for src_col in src_columns:
            src_data[src_col].append(match.second[src_col])
    data = dict(offset=offsets, ref_mag=ref_mags, src_mag=src_mags)
    data.update(src_data)
    data.update(ref_data)
    return pd.DataFrame(data=data)

def make_depth_map(df, nside=128, snr_bounds=None):
    """
    Make a map of depth in magnitudes.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame containing the reference catalog-matched point source
        information.
    nside: int [128]
        Healpix nside value.
    snr_bounds: (float, float) [None]
        Signal-to-noise bounds to bracket the desired n-sigma depth.
    """
    df = df.copy()
    df['snr'] = df['base_PsfFlux_instFlux']/df['base_PsfFlux_instFluxErr']
    if snr_bounds is not None:
        df = df.query(f'(snr >= {snr_bounds[0]}) and (snr <= {snr_bounds[1]})')
    pix_nums = hp.ang2pix(nside, [_.asDegrees() for _ in df['coord_ra']],
                          [_.asDegrees() for _ in df['coord_dec']], lonlat=True)
    map_out = np.zeros(12*nside**2)
    for pix in np.unique(pix_nums):
        mask = (pix == pix_nums)
        if np.count_nonzero(mask) > 0:
            map_out[pix] = np.nanmedian(df['ref_mag'][mask])
        else:
            map_out[pix] = 0.
    return map_out


def plot_binned_stats(x, values, x_range=None, bins=50, fmt='o', color='red'):
    """
    Plot the median of the values corresponding to the binned x values.
    Errors on the median are derived from the stdev of values.
    """
    binned_values = dict()
    for stat in ('median', 'std', 'count'):
        binned_values[stat], edges, _ \
            = binned_statistic(x, values, statistic=stat, range=x_range,
                               bins=bins)
    x_vals = (edges[1:] + edges[:-1])/2.
    yerr = binned_values['std']/np.sqrt(binned_values['count'])
    plt.errorbar(x_vals, binned_values['median'], yerr=yerr, fmt=fmt,
                 color=color)


def process_visit(butler, visit, center_radec=None, src_columns=None,
                  flux_type='base_PsfFlux'):
    """
    Perform point source matching on each dataref/sensor in a visit.
    """
    if src_columns is None:
        src_columns = ['coord_ra', 'coord_dec',
                       'base_SdssShape_xx', 'base_SdssShape_yy',
                       'base_PsfFlux_instFlux', 'base_PsfFlux_instFluxErr']
    datarefs = butler.subset('src', visit=visit)
    band = list(datarefs)[0].dataId['filter']
    ref_cats = RefCat(butler)
    if center_radec is None:
        # Try to obtain visit center from R22_S11.
        dataId = {'visit': visit, 'filter': band, 'raftName': 'R22',
                  'detectorName': 'S11'}
        centerCoord = ref_cats.ccd_center(dataId)
    else:
        centerCoord = lsst_geom.SpherePoint(center_radec[0]*lsst_geom.degrees,
                                            center_radec[1]*lsst_geom.degrees)
    print(centerCoord)
    radius = 2.1
    ref_cat = ref_cats(centerCoord, band, radius=radius)
    df = None
    for i, dataref in enumerate(datarefs):
        print(i, dataref.dataId)
        my_df = point_source_matches(dataref, ref_cat, src_columns=src_columns,
                                     flux_type=flux_type)
        if df is None:
            df = my_df
        else:
            df = df.append(my_df, ignore_index=True)
    return df
