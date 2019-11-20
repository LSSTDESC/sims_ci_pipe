"""
Module to create single frame processing validation plots.
"""
import os
from collections import defaultdict
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import pandas as pd
import healpy as hp
import lsst.afw.table as afw_table
import lsst.geom as lsst_geom
import lsst.daf.persistence as dp
from lsst.meas.algorithms import LoadIndexedReferenceObjectsTask


__all__ = ['RefCat', 'point_source_matches', 'visit_ptsrc_matches',
           'make_depth_map', 'plot_binned_stats', 'get_center_radec',
           'plot_detection_efficiency', 'get_ref_cat']


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
        self._ref_task = None

    @property
    def ref_task(self):
        if self._ref_task is None:
            refConfig = LoadIndexedReferenceObjectsTask.ConfigClass()
            refConfig.filterMap = {_: f'lsst_{_}_smeared' for _ in 'ugrizy'}
            self._ref_task = LoadIndexedReferenceObjectsTask(self.butler,
                                                             config=refConfig)
        return self._ref_task

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
                      (model_flag == False) &
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


def get_ref_cat(butler, visit, center_radec, radius=2.1):
    ref_cats = RefCat(butler)
    band = list(butler.subset('src', visit=visit))[0].dataId['filter']
    centerCoord = lsst_geom.SpherePoint(center_radec[0]*lsst_geom.degrees,
                                        center_radec[1]*lsst_geom.degrees)
    return ref_cats(centerCoord, band, radius)


def visit_ptsrc_matches(butler, visit, center_radec, src_columns=None,
                        flux_type='base_PsfFlux'):
    """
    Perform point source matching on each dataref/sensor in a visit.
    """
    if src_columns is None:
        src_columns = ['coord_ra', 'coord_dec',
                       'base_SdssShape_xx', 'base_SdssShape_yy',
                       'base_PsfFlux_instFlux', 'base_PsfFlux_instFluxErr']
    datarefs = butler.subset('src', visit=visit)
    ref_cat = get_ref_cat(butler, visit, center_radec)
    df = None
    for i, dataref in enumerate(datarefs):
        try:
            my_df = point_source_matches(dataref, ref_cat,
                                         src_columns=src_columns,
                                         flux_type=flux_type)
        except dp.butlerExceptions.NoResults:
            pass
        else:
            print(i, dataref.dataId)
            if df is None:
                df = my_df
            else:
                df = df.append(my_df, ignore_index=True)
    return df


def make_depth_map(df, nside=128, snr_bounds=None):
    """
    Make a map of depth in magnitudes.

    Parameters
    ----------
    df: pandas.DataFrame
        Data frame containing the reference catalog-matched point source
        information.
    nside: int [128]
        Healpix nside value. Sets the resolution of the map.
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

def get_center_radec(butler, visit, opsim_db=None):
    """
    Get the coordinates of the center (or median location) of the CCDs
    from the specified visit.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        Butler pointing to the repo with the calexps and src catalogs.
    visit: int
        Visit number.
    opsim_db: str [None]
        Opsim db file, e.g., the DESC version of minion_1016.

    Returns
    -------
    (float, float):  The RA, Dec in degrees of the center of the focal
        plane.  If the opsim_db file is given, then the DESC dithered
        pointing direction is returned.
    """
    if opsim_db is not None:
        conn = sqlite3.connect(opsim_db)
        df = pd.read_sql('select descDitheredRA, descDitheredDec '
                         'from Summary where '
                         f'obshistid={visit} limit 1', conn)
        return np.degrees(df.iloc[0].coord_ra), np.degrees(df.iloc[0].coord_dec)
    ref_cat = RefCat(butler)
    ras, decs = [], []
    for dataref in butler.subset('calexp', visit=visit):
        try:
            ccd_center = ref_cat.ccd_center(dataref.dataId)
        except dp.butlerExceptions.NoResults:
            pass
        else:
            ras.append(ccd_center.getLongitude())
            decs.append(ccd_center.getLatitude())
    return np.median(ras).asDegrees(), np.median(decs).asDegrees()


def plot_detection_efficiency(butler, visit, df, ref_cat, x_range=None,
                              bins=20, flux_type='base_PsfFlux', nside=4096):
    """Plot detection efficiency for stars."""
    # Gather ra, dec values for point sources in src catalogs.
    src_ra, src_dec = [], []
    band = None
    for dataref in butler.subset('src', visit=visit):
        try:
            src = dataref.get('src')
        except dp.butlerExceptions.NoResults:
            continue
        if band is None:
            band = dataref.dataId['filter']
        # Apply point source selections to the source catalog.
        ext = src.get('base_ClassificationExtendedness_value')
        model_flag = src.get(f'{flux_type}_flag')
        model_flux = src.get(f'{flux_type}_instFlux')
        num_children = src.get('deblend_nChild')
        coord_ra = src.get('coord_ra')
        coord_dec = src.get('coord_dec')
        index = ((ext == 0) &
                 (model_flag == False) &
                 (model_flux > 0) &
                 (num_children == 0))
        src_ra.extend(coord_ra[index])
        src_dec.extend(np.pi/2. - coord_dec[index])

    # ra, dec values for point sources in ref_cat.
    ref_ra = ref_cat.get('coord_ra')
    ref_dec = np.pi/2. - ref_cat.get('coord_dec')

    # Create a mask for all ref_cat point sources in healpixels that
    # have a detected point source from the src catalogs.
    pixnums = hp.ang2pix(nside, src_ra, src_dec, lonlat=False)
    pixnums_true = hp.ang2pix(nside, ref_ra, ref_dec, lonlat=False)
    ref_index = ((ref_cat.get('variable') == 0) &
                 (ref_cat.get('resolved') == 0) &
                 np.in1d(pixnums_true, np.unique(pixnums)))

    ref_mags0 = ref_cat.get(f'lsst_{band}')
    if x_range is None:
        x_range = min(ref_mags0), max(ref_mags0)

    src_count, edges, _ = binned_statistic(df['ref_mag'], df['ref_mag'],
                                           statistic='count', range=x_range,
                                           bins=bins)
    ref_mags = ref_mags0[ref_index]
    ref_count, edges, _ = binned_statistic(ref_mags, ref_mags,
                                           statistic='count', range=x_range,
                                           bins=bins)
    x_vals = (edges[1:] + edges[:-1])/2.
    y_vals = src_count/ref_count
    yerr = np.sqrt(src_count + ref_count)/ref_count
    plt.errorbar(x_vals, y_vals, yerr=yerr, fmt='.')
    plt.xlabel('ref_mag')
    plt.ylabel('Detection efficiency (stars)')
