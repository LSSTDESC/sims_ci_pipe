"""
Module to create a reference catalog derived from a star grid instance
catalog.
"""
import os
import sys
import lsst.utils
from lsst.sims.photUtils import BandpassDict
import desc.imsim

def sed_file_path(sed_file):
    return os.path.join(lsst.utils.getPackageDir('sims_sed_library'), sed_file)

def make_reference_catalog(instcat, outfile=None):
    bp_dict = BandpassDict.loadTotalBandpassesFromFiles()

visit = 169767
band = 'y'
infile = 'v{visit}-{band}_grid/star_grid_{visit}.txt'.format(**locals())
outfile = 'reference_catalog_v{visit}.txt'.format(**locals())

with open(infile, 'r') as fd, open(outfile, 'w') as output:
    output.write('# id ra dec sigma_ra sigma_dec ra_smeared dec_smeared u sigma_u g sigma_g r sigma_r i sigma_i z sigma_z y sigma_y u_smeared g_smeared r_smeared i_smeared z_smeared y_smeared u_rms g_rms r_rms i_rms z_rms y_rms isresolved isagn properMotionRa properMotionDec parallax radialVelocity\n')
    for i, line in enumerate(fd):
        tokens = line.split()
        id = tokens[1]
        ra = float(tokens[2])
        dec = float(tokens[3])
        mag_norm = float(tokens[4])
        gAv = float(tokens[-2])
        gRv = float(tokens[-1])
        sed_obj = desc.imsim.SedWrapper(sed_file, mag_norm, redshift, iAv, iRv,
                                        gAv, gRv, bp_dict).sed_obj
        u = sed_obj.calcMag(bp_dict['u'])
        g = sed_obj.calcMag(bp_dict['g'])
        r = sed_obj.calcMag(bp_dict['r'])
        i = sed_obj.calcMag(bp_dict['i'])
        z = sed_obj.calcMag(bp_dict['z'])
        y = sed_obj.calcMag(bp_dict['y'])
        output.write('{id}, {ra:.10f}, {dec:.10f}, 0.000000027778, 0.000000027778, {ra:.10f}, {dec:.10f}, {u:.5f}, 0.001, {g:.5f}, 0.001, {r:.5f}, 0.001, {i:.5f}, 0.001, {z:.5f}, 0.001, {y:.5f}, 0.001,  {u:.5f}, {g:.5f}, {r:.5f}, {i:.5f}, {z:.5f}, {y:.5f}, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0\n'.format(**locals()))
