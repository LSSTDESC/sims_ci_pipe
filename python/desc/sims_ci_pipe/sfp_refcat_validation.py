"""
Module to create single frame processing validation plots.
"""
import os
from collections import defaultdict
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.stats import binned_statistic
import pandas as pd
import healpy as hp
import lsst.afw.fits as afw_fits
import lsst.afw.table as afw_table
import lsst.geom as lsst_geom
import lsst.daf.butler as daf_butler
from lsst.meas.algorithms import LoadReferenceObjectsTask, \
    ReferenceObjectLoader
from .psf_mag_check import psf_mag_check
from .psf_whisker_plot import psf_whisker_plot
from .get_point_sources import get_band

__all__ = ['RefCat', 'point_source_matches', 'visit_ptsrc_matches',
           'make_depth_map', 'plot_binned_stats', 'get_center_radec',
           'plot_detection_efficiency', 'get_ref_cat',
           'sfp_validation_plots']


class RefCat:
    """
    Class to provide access to sky cone selected reference catalog data
    for DC2.
    """
    def __init__(self, butler, dstype='cal_ref_cat_2_2'):
        """
        Parameters
        ----------
        butler: lsst.daf.butler.Butler
            Butler pointing to the data repo with the processed visits.
        """
        self.butler = butler
        self.dstype = dstype
        self._ref_task = None

    @property
    def ref_task(self):
        """
        Handle for the ReferenceObjectsTask.
        """
        if self._ref_task is None:
            registry = self.butler.registry
            dsrefs = registry.queryDatasets(self.dstype)
            refCats = [daf_butler.DeferredDatasetHandle(self.butler, _, {})
                       for _ in dsrefs]
            dataIds = [registry.expandDataId(_.dataId) for _ in dsrefs]
            refConfig = LoadReferenceObjectsTask.ConfigClass()
            refConfig.filterMap = {_: f'lsst_{_}_smeared' for _ in 'ugrizy'}
            self._ref_task = ReferenceObjectLoader(dataIds=dataIds,
                                                   refCats=refCats,
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


def point_source_matches(dsref, butler, ref_cat0, max_offset=0.1,
                         src_columns=(), ref_columns=(),
                         flux_type='base_PsfFlux'):
    """
    Match point sources between a reference catalog and the dataset ref
    pointing to a src catalog.

    Parameters
    ----------
    dsref: lsst.daf.butler.DatasetRef
        DatasetRef pointing to the desired sensor-visit.
    butler: lsst.daf.butler.Butler
        Butler pointing to the data repo with the processed visits.
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
    src0 = butler.get('src', dataId=dsref.dataId)
    band = dsref.dataId['band']

    # Apply point source selections to the source catalog.
    ext = src0.get('base_ClassificationExtendedness_value')
    model_flag = src0.get(f'{flux_type}_flag')
    model_flux = src0.get(flux_col)
    num_children = src0.get('deblend_nChild')
    src = src0.subset((ext == 0) &
                      ~model_flag &
                      (model_flux > 0) &
                      (num_children == 0))

    # Match RA, Dec with the reference catalog stars.
    ref_cat = ref_cat0.subset((ref_cat0.get('resolved') == 0))
    radius = lsst_geom.Angle(max_offset, lsst_geom.arcseconds)
    matches = afw_table.matchRaDec(ref_cat, src, radius)
    num_matches = len(matches)

    offsets = np.zeros(num_matches, dtype=float)
    ref_ras = np.zeros(num_matches, dtype=float)
    ref_decs = np.zeros(num_matches, dtype=float)
    ref_mags = np.zeros(num_matches, dtype=float)
    src_mags = np.zeros(num_matches, dtype=float)
    ref_data = defaultdict(list)
    src_data = defaultdict(list)
    calib = butler.get('calexp', dataId=dsref.dataId).getPhotoCalib()
    for i, match in enumerate(matches):
        offsets[i] = np.degrees(match.distance)*3600*1000.
        ref_mags[i] = match.first[f'lsst_{band}']
        ref_ras[i] = match.first['coord_ra']
        ref_decs[i] = match.first['coord_dec']
        src_mags[i] = calib.instFluxToMagnitude(match.second[flux_col])
        for ref_col in ref_columns:
            ref_data[f'ref_{ref_col}'].append(match.first[ref_col])
        for src_col in src_columns:
            src_data[src_col].append(match.second[src_col])
    data = dict(offset=offsets, ref_mag=ref_mags, src_mag=src_mags,
                ref_ra=ref_ras, ref_dec=ref_decs)
    data.update(src_data)
    data.update(ref_data)
    return pd.DataFrame(data=data)


def get_ref_cat(butler, visit, band, center_radec, radius=2.1):
    """
    Get the reference catalog for the desired visit for the requested
    sky location and sky cone radius.
    """
    ref_cats = RefCat(butler)
    centerCoord = lsst_geom.SpherePoint(center_radec[0]*lsst_geom.degrees,
                                        center_radec[1]*lsst_geom.degrees)
    return ref_cats(centerCoord, band, radius)


def visit_ptsrc_matches(butler, visit, band, center_radec, src_columns=None,
                        max_offset=0.1, flux_type='base_PsfFlux'):
    """
    Perform point source matching on each dataref/sensor in a visit.
    """
    if src_columns is None:
        src_columns = ['coord_ra', 'coord_dec',
                       'base_SdssShape_xx', 'base_SdssShape_yy',
                       f'{flux_type}_instFlux', f'{flux_type}_instFluxErr']
    dsrefs = butler.registry.queryDatasets('src', visit=visit,
                                             findFirst=True)
    ref_cat = get_ref_cat(butler, visit, band, center_radec)
    df = None
    detectors = set()
    for i, dsref in enumerate(dsrefs):
        detector = dsref.dataId['detector']
        if detector in detectors:
            continue
        detectors.add(detector)
        try:
            my_df = point_source_matches(
                daf_butler.DeferredDatasetHandle(butler, dsref, None),
                butler,
                ref_cat,
                max_offset=max_offset,
                src_columns=src_columns,
                flux_type=flux_type)
        except afw_fits.FitsError:
            print("FitsError raised reading sfp data for", dsref.dataId)
            print(eobj)
        else:
            print(i, dsref.dataId)
            if df is None:
                df = my_df
            else:
                df = pd.concat([df, my_df], ignore_index=True)
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


def plot_binned_stats(x, values, x_range=None, bins=50, fmt='.', color='red',
                      skip_plots=False):
    """
    Plot the median of the values corresponding to the binned x values.
    Errors on the median are derived from the stdev of values.
    """
    binned_values = dict()
    for stat in ('median', 'std', 'count'):
        binned_values[stat], edges, _ \
            = binned_statistic(x, values, statistic=stat, range=x_range,
                               bins=bins)
    index = np.where(binned_values['count'] > 0)
    x_vals = (edges[1:] + edges[:-1])[index]/2.
    y_vals = binned_values['median'][index]
    yerr = binned_values['std'][index]/np.sqrt(binned_values['count'][index])
    if not skip_plots:
        plt.errorbar(x_vals, y_vals, yerr=yerr, fmt=fmt, color=color)
    return x_vals, y_vals, yerr


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
        # Get the pointing center from the opsim db file.
        conn = sqlite3.connect(opsim_db)
        df = pd.read_sql('select descDitheredRA, descDitheredDec '
                         'from Summary where '
                         f'obshistid={visit} limit 1', conn)
        return (np.degrees(df.iloc[0].descDitheredRA),
                np.degrees(df.iloc[0].descDitheredDec))

    # Return the center of detector=94, if it's available.
    ref_cat = RefCat(butler)
    dsrefs = butler.registry.queryDatasets('calexp', visit=visit,
                                             findFirst=True)
    dataId = dict()
    dataId.update(list(dsrefs)[0].dataId)
    dataId['detector'] = 94
    try:
        ccd_center = ref_cat.ccd_center(dataId)
    except Exception as eobj:
        print(eobj)
    else:
        return (ccd_center.getLongitude().asDegrees(),
                ccd_center.getLatitude().asDegrees())

    # Detector 94 isn't available, so loop over all available CCDs and use
    # the medians in ra, dec of the coordinate centers.
    ras, decs = [], []
    for dsref in dsrefs:
        try:
            ccd_center = ref_cat.ccd_center(dsref.dataId)
        except afw_fits.FitsError:
            print("FitsError raised reading sfp data for", dsref.dataId)
            print(eobj)
        except:
            pass
        else:
            ras.append(ccd_center.getLongitude())
            decs.append(ccd_center.getLatitude())
    return np.median(ras).asDegrees(), np.median(decs).asDegrees()


def plot_detection_efficiency(butler, visit, df, ref_cat, x_range=None,
                              y_range=(-0.2, 1.2), bins=20,
                              flux_type='base_PsfFlux', nside=4096,
                              color='blue'):
    """Plot detection efficiency for stars."""
    # Gather ra, dec values for point sources in src catalogs.
    src_ra, src_dec = [], []
    band = None
    dsrefs = butler.registry.queryDatasets('src', visit=visit,
                                             findFirst=True)
    for dsref in dsrefs:
        try:
            src = butler.get(dsref)
        except afw_fits.FitsError:
            print("FitsError raised reading sfp data for", dsref.dataId)
            print(eobj)
            continue
        if band is None:
            band = get_band(butler, dsref)
        # Apply point source selections to the source catalog.
        ext = src.get('base_ClassificationExtendedness_value')
        model_flag = src.get(f'{flux_type}_flag')
        model_flux = src.get(f'{flux_type}_instFlux')
        num_children = src.get('deblend_nChild')
        coord_ra = src.get('coord_ra')
        coord_dec = src.get('coord_dec')
        index = ((ext == 0) &
                 ~model_flag &
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
    index = np.where(ref_count > 0)
    x_vals = (edges[1:] + edges[:-1])[index]/2.
    y_vals = src_count[index]/ref_count[index]
    yerr = np.sqrt(src_count[index] + ref_count[index])/ref_count[index]
    plt.errorbar(x_vals, y_vals, yerr=yerr, fmt='.', color=color)
    plt.ylim(*y_range)
    plt.xlabel('ref_mag')
    plt.ylabel('Detection efficiency (stars)', color=color)


def get_five_sigma_depth(opsim_db_file, visit):
    """
    Get the predicted 5-sigma depth for the specified visit from the
    Opsim sb Summary table.
    """
    conn = sqlite3.connect(opsim_db_file)
    query = ('select fiveSigmaDepth from Summary where '
             f'obsHistID={visit} limit 1')
    return pd.read_sql(query, conn).iloc[0].fiveSigmaDepth


def extrapolate_nsigma(ref_mag, SNR, nsigma=5):
    """
    Fit a parabola to log10(SNR) vs ref_mag and
    extrapolate/interpolate to find the n-sigma magnitude limit.
    """
    log10_SNR = np.log10(SNR)
    index = np.where(log10_SNR == log10_SNR)
    pars = np.polyfit(log10_SNR[index], ref_mag[index], 2)
    mag = np.poly1d(pars)
    return mag(np.log10(nsigma)), mag


def plot_dmags(psf_mags, ref_mags, xlabel='psf_mag - ref_mag', sn_min=150,
               dmag_range=(-0.05, 0.05), skip_plots=False):
    """
    Plot delta mag distribution.

    Parameters
    ----------
    psf_mags: np.array
        base_PsfFlux magnitudes.
    ref_mags: np.array
        Reference catalog magnitudes.
    xlabel: str ['psf_mag - ref_mag']
        Label for x-axis.
    sn_min: float [150]
        Signal-to-noise minimum applied to psfFlux/psfFluxErr.
    dmag_range: (float, float) [(-0.05, 0.05)]
        Plotting range in magnitudes.

    Returns
    -------
    float: median(psf_mags - ref_mags)
    """
    delta_mag = psf_mags - ref_mags
    dmag_median = np.median(delta_mag)
    if skip_plots:
        return dmag_median
    plt.hist(delta_mag, range=dmag_range, bins=100, histtype='step',
             density=True)
    plt.axvline(0, linestyle=':')
    plt.axvline(dmag_median, linestyle='--')
    plt.annotate(f'median: {dmag_median*1000:.2f} mmag\n'
                 f'psfFlux/psfFluxErr > {sn_min}', (0.05, 0.95),
                 xycoords='axes fraction', verticalalignment='top')
    plt.xlabel(xlabel)
    return dmag_median


def sfp_validation_plots(repo, visit, outdir='.', flux_type='base_PsfFlux',
                         opsim_db=None, figsize=(12, 10), max_offset=0.1,
                         sn_min=150, instrument='LSSTCam-imSim',
                         collections=None, skip_plots=False):
    """
    Create the single-frame validation plots.

    Parameters
    ----------
    repo: str
        Data repository containing calexps.
    visit: int
        Visit number.
    outdir: str ['.']
        Directory to contain output files.
    flux_type: str ['base_PsfFlux']
        Flux column to use for selecting well-measured point sources.
    opsim_db: str [None]
        OpSim db file containing pointing information.  This is used
        to get the pointing direction for the ref cat selection and
        the predicted five sigma depth for the visit.  If None, then
        the pointing direction will be inferred from the calexps.
    figsize: (float, float) [(12, 10)]
        Size of the figure in inches.
    max_offset: float [0.1]
        Maximum offset, in arcsec, for positional matching of point
        sources to ref cat stars.
    sn_min: float [150]
        Mininum signal-to-noise cut on psfFlux/psfFluxErr.

    Returns
    -------
    pandas.DataFrame containg the visit-level metrics:
        (median astrometric offset, median delta magitude, median T value,
         extrapolated five sigma depth)
    """
    if collections is None:
        butler = daf_butler.Butler(repo)
        collections = list(butler.registry.queryCollections())

    butler = daf_butler.Butler(repo, collections=collections)

    try:
        dsrefs = list(butler.registry.queryDatasets('src', visit=visit,
                                                    instrument=instrument,
                                                    collections=collections,
                                                    findFirst=True))
        band = get_band(butler, dsrefs[0])
    except Exception as eobj:
        print('visit:', visit)
        print(eobj)
        raise eobj
    center_radec = get_center_radec(butler, visit, opsim_db)
    ref_cat = get_ref_cat(butler, visit, band, center_radec)

    os.makedirs(outdir, exist_ok=True)
    pickle_file = os.path.join(outdir, f'sfp_validation_v{visit}-{band}.pkl')

    if not os.path.isfile(pickle_file):
        df = visit_ptsrc_matches(butler, visit, band, center_radec,
                                 max_offset=max_offset, flux_type=flux_type)
        df.to_pickle(pickle_file)
    else:
        df = pd.read_pickle(pickle_file)

    median_offset = np.median(df['offset'])
    if not skip_plots:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(2, 2, 1)

        coord_ra = np.array([_.asRadians() for _ in df['coord_ra']])
        coord_dec = np.array([_.asRadians() for _ in df['coord_dec']])
        dra = (np.degrees((df['ref_ra'] - coord_ra)*np.cos(df['ref_dec']))
               *3600*1000)
        ddec = np.degrees((df['ref_dec'] - coord_dec))*3600*1000

        max_offset *= 1e3*1.2
        xy_range = (-max_offset, max_offset)
        plt.hexbin(dra, ddec, mincnt=1)
        plt.xlabel('RA offset (mas)')
        plt.ylabel('Dec offset (mas)')
        plt.xlim(*xy_range)
        plt.ylim(*xy_range)

        nullfmt = NullFormatter()

        ax_ra = ax.twinx()
        ax_ra.yaxis.set_major_formatter(nullfmt)
        ax_ra.yaxis.set_ticks([])
        bins, _, _ = plt.hist(dra, bins=50, histtype='step', range=xy_range,
                              density=True, color='red')
        ax_ra.set_ylim(0, 2.3*np.max(bins))

        ax_dec = ax.twiny()
        ax_dec.xaxis.set_major_formatter(nullfmt)
        ax_dec.xaxis.set_ticks([])
        bins, _, _ = plt.hist(ddec, bins=50, histtype='step', range=xy_range,
                              density=True, color='red',
                              orientation='horizontal')
        ax_dec.set_xlim(0, 2.3*np.max(bins))

        plt.annotate(f'{median_offset:.1f} mas median offset', (0.5, 0.95),
                     xycoords='axes fraction', horizontalalignment='left')

        plt.title(f'v{visit}-{band}')
        plt.colorbar()

    if not skip_plots:
        fig.add_subplot(2, 2, 2)
        bins = 20
        delta_mag = df['src_mag'] - df['ref_mag']
        dmag_med = np.nanmedian(delta_mag)
        ymin, ymax = dmag_med - 0.5, dmag_med + 0.5
        plt.hexbin(df['ref_mag'], delta_mag, mincnt=1)
        plot_binned_stats(df['ref_mag'], delta_mag, x_range=plt.axis()[:2],
                          bins=20)
        plt.xlabel('ref_mag')
        plt.ylabel(f'{flux_type}_mag - ref_mag')
        plt.ylim(ymin, ymax)
        plt.title(f'v{visit}-{band}')
        plt.colorbar()
        xmin, xmax = plt.axis()[:2]

    T = (df['base_SdssShape_xx'] + df['base_SdssShape_yy'])*0.2**2
    tmed = np.nanmedian(T)
    if not skip_plots:
        fig.add_subplot(2, 2, 3)
        ymin, ymax = tmed - 0.1, tmed + 0.1
        plt.hexbin(df['ref_mag'], T, mincnt=1, extent=(xmin, xmax, ymin, ymax))
        plot_binned_stats(df['ref_mag'], T, x_range=plt.axis()[:2], bins=20)
        plt.xlabel('ref_mag')
        plt.ylabel('T (arcsec**2)')
        plt.ylim(ymin, ymax)
        plt.title(f'v{visit}-{band}')
        plt.colorbar()

    x_range = (12, 26)
    if not skip_plots:
        ax1 = fig.add_subplot(2, 2, 4)
        plot_detection_efficiency(butler, visit, df, ref_cat, x_range=x_range)
        plt.title(f'v{visit}-{band}')

        ax2 = ax1.twinx()
        ax2.set_ylabel('S/N', color='red')

    snr = df[f'{flux_type}_instFlux']/df[f'{flux_type}_instFluxErr']
    ref_mags, SNR_values, _ = plot_binned_stats(df['ref_mag'], snr,
                                                x_range=x_range, bins=20,
                                                color='red',
                                                skip_plots=skip_plots)
    m5, mag_func = extrapolate_nsigma(ref_mags, SNR_values, nsigma=5)
    if not skip_plots:
        plt.xlim(*x_range)

        plt.yscale('log')
        ymin, ymax = 1, plt.axis()[-1]
        plt.ylim(ymin, ymax)
        plt.axhline(5, linestyle=':', color='red')
        yvals = np.logspace(np.log10(ymin), np.log10(ymax), 50)
        plt.plot(mag_func(np.log10(yvals)), yvals, linestyle=':', color='red')
        if opsim_db is not None:
            plt.axvline(get_five_sigma_depth(opsim_db, visit),
                        linestyle='--', color='red')

        plt.tight_layout()
        outfile = os.path.join(outdir, f'sfp_validation_v{visit}-{band}.png')
        plt.savefig(outfile)

    # Make plot of psf_mag - calib_mag distribution.
    if not skip_plots:
        fig = plt.figure(figsize=(6, 4))
    dmag_calib_median = psf_mag_check(repo, visit, sn_min=sn_min)
    if not skip_plots:
        plt.title(f'v{visit}-{band}')
        outfile = os.path.join(outdir, f'delta_mag_calib_v{visit}-{band}.png')
        plt.savefig(outfile)

    # Make plot of psf_mag - ref_mag distribution.
    my_df = df.query(f'{flux_type}_instFlux/{flux_type}_instFluxErr'
                     f' > {sn_min}')
    if not skip_plots:
        fig = plt.figure(figsize=(6, 4))
    dmag_ref_median = plot_dmags(my_df['src_mag'], my_df['ref_mag'],
                                 sn_min=sn_min, skip_plots=skip_plots)
    if not skip_plots:
        plt.title(f'v{visit}-{band}')
        outfile = os.path.join(outdir, f'delta_mag_ref_v{visit}-{band}.png')
        plt.savefig(outfile)

        # Make psf whisker plot.
        psf_whisker_plot(butler, visit)
        outfile = os.path.join(outdir, f'psf_whisker_plot_v{visit}-{band}.png')
        plt.savefig(outfile)

    df = pd.DataFrame(data=dict(visit=[visit], ast_offset=[median_offset],
                                dmag_ref_median=[dmag_ref_median],
                                dmag_calib_median=[dmag_calib_median],
                                T_median=[tmed], m5=[m5], band=[band]))
    metrics_file = os.path.join(outdir, f'sfp_metrics_v{visit}-{band}.pkl')
    df.to_pickle(metrics_file)

    return df
