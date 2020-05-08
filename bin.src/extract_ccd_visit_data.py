#!/usr/bin/env python
"""
Script to extract CCD-level data, such as seeing, sky level, zero-points,
m5 value from single visit data.
"""
from collections import defaultdict
import multiprocessing
import numpy as np
import pandas as pd
import lsst.afw.math as afwMath
import lsst.daf.persistence as dp


def make_stats(image, flags):
    """Succinct wrapper function for makeStatistics."""
    return afwMath.makeStatistics(image, flags).getValue()


def extract_visit_data(butler, visit, visit_name='visit'):
    """
    Extract CCD-level data for a single visit.
    """
    datarefs = butler.subset('calexp', dataId={visit_name: visit})
    print(f'processing {len(datarefs)} CCDs from {visit}')
    return extract_ccd_data(datarefs, visit_name=visit_name)


def extract_ccd_data(datarefs, visit_name='visit'):
    """
    Extract CCD-level data for a list of visit-level, single CCD datarefs.

    Returns
    -------
    pandas.DataFrame containing the CCD visit data.
    """
    data = defaultdict(list)
    for dataref in datarefs:
        try:
            calexp = dataref.get('calexp')
            calexp_bg = dataref.get('calexpBackground')
            src = dataref.get('src')
        except Exception as eobj:
            print(dataref.dataId, eobj)
            continue

        data['visit'].append(dataref.dataId[visit_name])
        data['band'].append(dataref.dataId['filter'])
        data['raft'].append(dataref.dataId['raftName'])
        data['sensor'].append(dataref.dataId['detectorName'])

        calib = calexp.getPhotoCalib()
        psf = calexp.getPsf()
        pixel_scale = calexp.getWcs().getPixelScale().asArcseconds()

        data['zero_point'].append(calib.instFluxToMagnitude(1))
        data['seeing'].append(psf.computeShape().getDeterminantRadius()
                              *2.35*pixel_scale)

        data['sky_level'].append(make_stats(calexp_bg[0][0].getStatsImage(),
                                            afwMath.MEDIAN))
        data['noise'].append(make_stats(calexp.getMaskedImage(),
                                        afwMath.STDEVCLIP))

        ext = src.get('base_ClassificationExtendedness_value')
        model_flag = src.get('base_PsfFlux_flag')
        model_flux = src.get('base_PsfFlux_instFlux')
        num_children = src.get('deblend_nChild')
        data['num_galaxies'].append(len(src.subset((ext != 0) &
                                                   (model_flag == False) &
                                                   (model_flux > 0) &
                                                   (num_children == 0))))
        stars = src.subset((ext == 0) &
                           (model_flag == False) &
                           (model_flux > 0) &
                           (num_children == 0)).copy(deep=True)
        data['num_stars'].append(len(stars))
        flux = stars.get('base_PsfFlux_instFlux')
        snr = flux/stars.get('base_PsfFlux_instFluxErr')
        index = np.where(snr < 10)
        flux_func = np.poly1d(np.polyfit(snr[index], flux[index], 2))
        data['m5'].append(calib.instFluxToMagnitude(flux_func(5)))

    return pd.DataFrame(data=data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Script to extract visit-level data for each CCD.")
    parser.add_argument('repo', type=str,
                        help='Repository with single frame processing data.')
    parser.add_argument('outfile', type=str, help='Output filename.')
    parser.add_argument('--visits', type=int, nargs='+',
                        help='Visits to process. If omitted, then visits '
                        'in the repo will be processed.')
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of subprocesses to use.')
    args = parser.parse_args()

    butler = dp.Butler(args.repo)
    visits = args.visits

    # Determine visit keyword name from the dataIds for this version
    # of the Stack.  Newer versions use `expId`; older ones use `visit`.
    if 'expId' in list(butler.subset('calexp'))[0].dataId:
        visit_name = 'expId'
    else:
        visit_name = 'visit'

    if not visits:
        # Get a sorted list of all visits in this repo.
        datarefs = butler.subset('calexp')
        visits = sorted(list(set([_.dataId[visit_name] for _ in datarefs])))

    if args.processes == 1:
        dfs = [extract_visit_data(butler, visit, visit_name=visit_name)
               for visit in visits]
    else:
        with multiprocessing.Pool(processes=processes) as pool:
            workers = [pool.apply_async(extract_visit_data,
                                        (butler, visit),
                                        dict(visit_name=visit_name))]
            pool.close()
            pool.join()
            dfs = [worker.get() for worker in workers]
    df = pd.concat(dfs)
    df.to_pickle(args.outfile)
