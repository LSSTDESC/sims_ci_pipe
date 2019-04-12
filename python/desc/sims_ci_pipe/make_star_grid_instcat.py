"""
Module to create an instance catalog of stars on a grid with a range of
mag_norm values derived from a simulated visit.
"""
import os
import sys
import warnings
import numpy as np
import lsst.daf.persistence as dp
import lsst.utils
from lsst.sims.photUtils import BandpassDict
import desc.imsim

__all__ = ['make_star_grid_instcat', 'make_reference_catalog']

def shuffled_mags(star_cat, mag_range=(16.3, 21)):
    """
    Extract the mag_norm values from star_dat and shuffle them
    so that they can be sampled without replacement.

    Parameters
    ----------
    star_cat: str
        phosim-style instance catalog of stars.
    mag_range: tuple [(16.3, 21)]
        Selection range of mag_norm values.

    Returns
    -------
    np.array of mag_norm values.
    """
    mags = []
    with desc.imsim.fopen(star_cat, 'rt') as fd:
        for line in fd:
            tokens = line.split()
            mag = float(tokens[4])
            if mag > mag_range[0] and mag < mag_range[1]:
                mags.append(mag)
    np.random.shuffle(mags)
    return np.array(mags)

def parse_instcat(instcat):
    """
    Parse the phosim_cat_*.txt instance catalog and return the
    star catalog filename, assuming it starts with `star_`, the visit,
    and the band.

    Parameters
    ----------
    instcat: str
        The phosim_cat_*.txt instance catalog.

    Returns
    -------
    tuple of star catalog full path, visit, band
    """
    instcat_dir = os.path.dirname(os.path.abspath(instcat))
    with open(instcat) as fd:
        for line in fd:
            if line.startswith('includeobj'):
                cat_name = line.strip().split()[-1]
                if cat_name.startswith('star_'):
                    star_cat = cat_name
            if line.startswith('obshistid'):
                visit = int(line.strip().split()[-1])
            if line.startswith('filter'):
                band = 'ugrizy'[int(line.strip().split()[-1])]
    return os.path.join(instcat_dir, star_cat), visit, band

def write_phosim_cat(instcat, outdir, star_grid_cat):
    """
    Write an updated version of the phosim_cat file with the star_grid_cat
    file as the only includeobj target.

    Parameters
    ----------
    instcat: str
        phosim_cat_*.txt file
    outdir: str
        output directory for new instance catalog
    star_grid_cat: str
        file name of star_cat file for new instance catalog.

    Returns
    -------
    None
    """
    outfile = os.path.join(outdir, os.path.basename(instcat))
    with open(instcat, 'r') as fd, open(outfile, 'w') as output:
        for line in fd:
            if not line.startswith('includeobj'):
                output.write(line)
        output.write('includeobj {}\n'.format(star_grid_cat))

def make_star_grid_instcat(butler, instcat, detectors=None, x_pixels=None,
                           y_pixels=None, mag_range=(16.3, 21)):
    """
    Create an instance catalog consisting of grids of stars on each chip,
    using the raw files and instance catalog from a simulated visit to
    provide the WCS info per CCD and visit information.

    Parameters
    ----------
    butler: lsst.daf.persistence.Butler
        The butler pointing to the repo with the desired raw files.
    instcat: str
        The instance catalog corresponding to the desired visit.
    detectors: sequence of ints [None]
        The detectors to process. If None, then use range(189).
    x_pixels: sequence of ints [None]
        The pixel coordinates in the x (i.e., serial)-direction. If None,
        then use np.linspace(150, 4050, 40).
    y_pixels: sequence of ints [None]
        The pixel coordinates in the y (i.e., parallel)-direction. If None,
        then use np.linspace(100, 3900, 39).
    mag_range: tuple [(16.3, 21)]
        Range of mag_norm values to sample from the input star_cat file.

    Returns
    -------
    list of missing detectors for the specified visit.
    """
    if detectors is None:
        detectors = range(189)
    if x_pixels is None:
        x_pixels = np.linspace(150, 4050, 40)
    if y_pixels is None:
        y_pixels = np.linspace(100, 3900, 39)

    template = "object {id} {ra:.15f} {dec:.15f} {mag:.8f} starSED/phoSimMLT/lte037-5.5-1.0a+0.4.BT-Settl.spec.gz 0 0 0 0 0 0 point none CCM 0.04056722 3.1\n"

    star_cat, visit, band = parse_instcat(instcat)

    num_stars = len(x_pixels)*len(y_pixels)
    mags = shuffled_mags(star_cat)[:num_stars]
    mags.sort()

    outdir = 'v{visit}-{band}_grid'.format(**locals())
    os.makedirs(outdir, exists_ok=True)

    star_grid_cat = 'star_grid_{visit}.txt'.format(**locals())
    write_phosim_cat(instcat, outdir, star_grid_cat)

    missing_detectors = []

    outfile = os.path.join(outdir, star_grid_cat)
    with open(outfile, 'w') as output:
        id = 0
        for detector in detectors:
            print("processing", detector)
            dataId = dict(visit=visit, detector=detector)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                try:
                    wcs = butler.get('raw', dataId=dataId).getWcs()
                except Exception as eobj:
                    print(eobj)
                    missing_detectors.append(detector)
            for x_pix in x_pixels:
                for y_pix in y_pixels:
                    ra, dec = [_.asDegrees() for _ in
                               wcs.pixelToSky(x_pix, y_pix)]
                    mag = mags[id % num_stars]
                    output.write(template.format(**locals()))
                    id += 1
    return missing_detectors

def sed_file_path(sed_file):
    return os.path.join(lsst.utils.getPackageDir('sims_sed_library'), sed_file)

def make_reference_catalog(instcat, outfile=None):
    bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
    star_cat, visit, band = parse_instcat(instcat)

    if outfile is None:
        outfile = 'reference_catalog_v{visit}.txt'.format(**locals())

    # Assume we are dealing only with stars, so set internal
    # extinction to zero.
    iAv, iRv = 0, 3.1

    with desc.imsim.fopen(star_cat, 'rt') as fd, open(outfile, 'w') as output:
        output.write('# id ra dec sigma_ra sigma_dec ra_smeared dec_smeared u sigma_u g sigma_g r sigma_r i sigma_i z sigma_z y sigma_y u_smeared g_smeared r_smeared i_smeared z_smeared y_smeared u_rms g_rms r_rms i_rms z_rms y_rms isresolved isagn properMotionRa properMotionDec parallax radialVelocity\n')
        for i, line in enumerate(fd):
            tokens = line.split()
            id = tokens[1]
            ra = float(tokens[2])
            dec = float(tokens[3])
            mag_norm = float(tokens[4])
            sed_file = sed_file_path(tokens[5])
            gAv = float(tokens[-2])
            gRv = float(tokens[-1])
            sed_obj = desc.imsim.SedWrapper(sed_file, mag_norm, redshift,
                                            iAv, iRv, gAv, gRv, bp_dict).sed_obj
            u = sed_obj.calcMag(bp_dict['u'])
            g = sed_obj.calcMag(bp_dict['g'])
            r = sed_obj.calcMag(bp_dict['r'])
            i = sed_obj.calcMag(bp_dict['i'])
            z = sed_obj.calcMag(bp_dict['z'])
            y = sed_obj.calcMag(bp_dict['y'])
            output.write('{id}, {ra:.10f}, {dec:.10f}, 0.000000027778, 0.000000027778, {ra:.10f}, {dec:.10f}, {u:.5f}, 0.001, {g:.5f}, 0.001, {r:.5f}, 0.001, {i:.5f}, 0.001, {z:.5f}, 0.001, {y:.5f}, 0.001,  {u:.5f}, {g:.5f}, {r:.5f}, {i:.5f}, {z:.5f}, {y:.5f}, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0\n'.format(**locals()))
