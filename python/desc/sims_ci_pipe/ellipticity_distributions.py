import numpy as np
import matplotlib.pyplot as plt
import lsst.daf.persistence as dp
from opsim_db_interface import OpSimDb

__all__ = ['plot_ellipticities']

def asymQ(ixx, iyy, ixy):
    asymQx = ixx - iyy
    asymQy = 2*ixy
    return np.sqrt(asymQx**2 + asymQy**2)


def trQ(ixx, iyy):
    return ixx + iyy


def get_a(ixx, iyy, ixy):
    return np.sqrt(0.5*(trQ(ixx, iyy) + asymQ(ixx, iyy, ixy)))


def get_b(ixx, iyy, ixy):
    return np.sqrt(0.5*(trQ(ixx, iyy) - asymQ(ixx, iyy, ixy)))


def get_e(ixx, iyy, ixy):
    a = get_a(ixx, iyy, ixy)
    b = get_b(ixx, iyy, ixy)
    return (a**2 - b**2)/(a**2 + b**2)


def get_point_sources(src, flux_type='base_PsfFlux'):
    ext = src.get('base_ClassificationExtendedness_value')
    model_flag = src.get(f'{flux_type}_flag')
    model_flux = src.get(f'{flux_type}_instFlux')
    num_children = src.get('deblend_nChild')
    return src.subset((ext == 0) &
                      (model_flag == False) &
                      (model_flux > 0) &
                      (num_children == 0))


def plot_ellipticities(butler, visits, opsim_db_file=None, min_altitude=80.,
                       seeing_range=(0.6, 0.8), e_range=(0, 0.1), bins=100):
    opsim_db = OpSimDb(opsim_db_file)
    ellipticities = []
    for visit in visits:
        row = opsim_db(visit)
#        if (np.degrees(row.altitude) < min_altitude or
#            not (seeing_range[0] < row.FWHMgeom < seeing_range[1])):
#            continue
        datarefs = butler.subset('src', visit=visit)
        for i, dataref in enumerate(datarefs):
            try:
                src = get_point_sources(dataref.get('src'))
            except dp.butlerExceptions.NoResults:
                continue
            for record in src:
                ellipticities.append(get_e(record['base_SdssShape_xx'],
                                           record['base_SdssShape_yy'],
                                           record['base_SdssShape_xy']))
    plt.hist(ellipticities, range=e_range, bins=bins)

if __name__ == '__main__':
    plt.ion()
    opsim_db_file = '/home/DC2/minion_1016_desc_dithered_v4.db'
    repo = '/home/Run2.2i/image_spot_check/repo/rerun/2019-11-19'
    butler = dp.Butler(repo)
    visits = set([_.dataId['visit'] for _ in butler.subset('src', filter='i')])
    plt.figure()
    plot_ellipticities(butler, visits, opsim_db_file=opsim_db_file)
