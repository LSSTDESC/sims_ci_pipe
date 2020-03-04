"""
Module to create an instance catalog of stars on a grid with a range of
mag_norm values derived from a simulated visit.
"""
import os
from collections import OrderedDict
import numpy as np
import lsst.utils
from lsst.sims.photUtils import BandpassDict
from lsst.sims.GalSimInterface import LSSTCameraWrapper
from lsst.sims.GalSimInterface.wcsUtils import tanSipWcsFromDetector
import lsst.obs.lsst as obs_lsst
import desc.imsim


__all__ = ['make_star_grid_instcat', 'make_reference_catalog']


camera = obs_lsst.imsim.ImsimMapper().camera
det_name = OrderedDict()
for i, det in enumerate(camera):
    det_name[i] \
        = 'R:{},{} S:{},{}'.format(*[_ for _ in det.getName() if _.isdigit()])

camera_wrapper = LSSTCameraWrapper()


def shuffled_mags(star_cat, mag_range=(16.3, 21)):
    """
    Extract the mag_norm values from star_cat and shuffle them
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
    with desc.imsim.fopen(star_cat, mode='rt') as fd:
        for line in fd:
            tokens = line.split()
            mag = float(tokens[4])
            if mag_range[0] < mag < mag_range[1]:
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
    full path to the phosim_cat file.
    """
    outfile = os.path.join(outdir, os.path.basename(instcat))
    with open(instcat, 'r') as fd, open(outfile, 'w') as output:
        for line in fd:
            if (not line.startswith('includeobj')
                and not line.startswith('minsource')):
                output.write(line)
        output.write('includeobj {}\n'.format(star_grid_cat))
    return os.path.abspath(outfile)


def make_star_grid_instcat(instcat, detectors=None, x_pixels=None,
                           y_pixels=None, mag_range=(16.3, 21),
                           max_x_offset=0, max_y_offset=0,
                           y_stagger=4, outdir=None, sorted_mags=False):
    """
    Create an instance catalog consisting of grids of stars on each
    chip, using the instance catalog from a simulated visit to provide
    the info for constructing the WCS per CCD.

    Parameters
    ----------
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
    max_x_offset: float [0]
        Maximum offset in pixels to be drawn in the x-direction to
        displace each star from its nominal grid position.  These
        offsets helps prevent failures in the astrometric solution
        that arises from trying to match to a regular grid of
        reference stars.
    max_y_offset: float [0]
        Maximum offset in pixels to be drawn in the y-direction to
        displace each star from its nominal grid position.  These
        offsets helps prevent failures in the astrometric solution
        that arises from trying to match to a regular grid of
        reference stars.
    y_stagger: int [4]
        Stagger rows by y_step/y_stagger*(ix % ystagger)
    outdir: str [None]
        Output directory for instance catalog files.  If None, then use
        f'v{visit}-{band}_grid'.
    sorted_mags: bool [False]
        Flag to sort magnitudes so that brightest objects are at the bottom
        of the CCD.

    Returns
    -------
    The full path to the star grid instance catalog.
    """
    if detectors is None:
        detectors = range(189)
    if x_pixels is None:
        x_pixels = np.linspace(200, 3800, 36)
    if y_pixels is None:
        y_pixels = np.linspace(200, 3800, 36)

    template = "object {my_id} {ra:.15f} {dec:.15f} {mag:.8f} starSED/phoSimMLT/lte037-5.5-1.0a+0.4.BT-Settl.spec.gz 0 0 0 0 0 0 point none CCM 0.04056722 3.1\n"

    star_cat, visit, band = parse_instcat(instcat)
    obs_md \
        = desc.imsim.phosim_obs_metadata(desc.imsim.metadata_from_file(instcat))

    num_stars = len(x_pixels)*len(y_pixels)
    mags = shuffled_mags(star_cat, mag_range=mag_range)[:num_stars]
    if sorted_mags:
        mags.sort()

    if outdir is None:
        outdir = f'v{visit}-{band}_grid'
    os.makedirs(outdir, exist_ok=True)

    star_grid_cat = 'star_grid_{visit}.txt'.format(**locals())
    phosim_cat_file = write_phosim_cat(instcat, outdir, star_grid_cat)

    outfile = os.path.join(outdir, star_grid_cat)
    y_step = y_pixels[1] - y_pixels[0]
    with open(outfile, 'w') as output:
        my_id = 0
        for detector in detectors:
            print("processing", detector)
            wcs = tanSipWcsFromDetector(det_name[detector], camera_wrapper,
                                        obs_md, epoch=2000.)
            if not sorted_mags:
                # Re-shuffle the magnitudes for each CCD.
                np.random.shuffle(mags)
            for ix, x_pix in enumerate(x_pixels):
                # Stagger rows by quarter steps
                y_offset = y_step/y_stagger*(ix % y_stagger)
                for y_pix in y_pixels:
                    dx = np.random.uniform(high=max_x_offset)
                    dy = np.random.uniform(high=max_y_offset)
                    ra, dec = [_.asDegrees() for _ in
                               wcs.pixelToSky(x_pix + dx,
                                              y_pix + dy + y_offset)]
                    mag = mags[my_id % num_stars]
                    output.write(template.format(**locals()))
                    my_id += 1
    return phosim_cat_file


def sed_file_path(sed_file):
    """Return the path to the SED file."""
    return os.path.join(lsst.utils.getPackageDir('sims_sed_library'), sed_file)


def make_reference_catalog(instcat, outfile=None):
    """
    Make a reference catalog corresponding to an instance catalog of stars.

    Parameters
    ----------
    instcat: str
        Instance catalog to provide the reference catalog objects.

    Returns
    -------
    The full path to the reference catalog that was created.
    """
    bp_dict = BandpassDict.loadTotalBandpassesFromFiles()
    star_cat, visit, band = parse_instcat(instcat)

    if outfile is None:
        outfile = 'reference_catalog_v{visit}-{band}.txt'.format(**locals())

    # Assume we are dealing only with stars, so set redshift and internal
    # extinction to zero.
    redshift = 0
    iAv, iRv = 0, 3.1

    with desc.imsim.fopen(star_cat, mode='rt') as fd, \
         open(outfile, 'w') as output:
        output.write('# id ra dec sigma_ra sigma_dec ra_smeared dec_smeared u sigma_u g sigma_g r sigma_r i sigma_i z sigma_z y sigma_y u_smeared g_smeared r_smeared i_smeared z_smeared y_smeared u_rms g_rms r_rms i_rms z_rms y_rms isresolved isagn properMotionRa properMotionDec parallax radialVelocity\n')
        for i, line in enumerate(fd):
            tokens = line.split()
            my_id = tokens[1]
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
            output.write('{my_id}, {ra:.10f}, {dec:.10f}, 0.000000027778, 0.000000027778, {ra:.10f}, {dec:.10f}, {u:.5f}, 0.001, {g:.5f}, 0.001, {r:.5f}, 0.001, {i:.5f}, 0.001, {z:.5f}, 0.001, {y:.5f}, 0.001,  {u:.5f}, {g:.5f}, {r:.5f}, {i:.5f}, {z:.5f}, {y:.5f}, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0\n'.format(**locals()))

    return os.path.abspath(outfile)
