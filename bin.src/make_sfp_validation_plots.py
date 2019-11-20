#!/usr/bin/env python
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import healpy as hp
import lsst.daf.persistence as dp
import desc.sims_ci_pipe as scp

parser = argparse.ArgumentParser()
parser.add_argument('repo', help='Data repo with calexps')
parser.add_argument('visit', type=int, help='visit to process')
parser.add_argument('--pickle_file', type=str, default=None,
                    help='Pickle file to hold matched point source catalog')
parser.add_argument('--outfile', type=str, default=None,
                    help='Name of png file to contain plots')
parser.add_argument('--flux_type', type=str, default='base_PsfFlux',
                    help='flux type of point sources to use')
parser.add_argument('--opsim_db', type=str, default=None,
                    help='opsim db to use for getting visit info')
args = parser.parse_args()

visit = 214432
band = 'i'
outfile = f'sfp_validation_data_{args.visit}.pkl'
flux_type = 'base_PsfFlux'
butler = dp.Butler(args.repo)
center_radec = scp.get_center_radec(butler, visit, args.opsim_db)
ref_cat = scp.get_ref_cat(butler, visit, center_radec)

if not os.path.isfile(args.pickle_file):
    df = scp.visit_ptsrc_matches(butler, visit, center_radec)
    df.to_pickle(args.pickle_file)
else:
    df = pd.read_pickle(args.pickle_file)

fig = plt.figure(figsize=(16, 16))
fig.add_subplot(2, 2, 1)
plt.hist(df['offset'], bins=40)
plt.xlabel('offset (mas)')
plt.title(f'v{visit}-{band}')

fig.add_subplot(2, 2, 2)
bins = 20
delta_mag = df['src_mag'] - df['ref_mag']
plt.hexbin(df['ref_mag'], delta_mag, mincnt=1)
scp.plot_binned_stats(df['ref_mag'], delta_mag, x_range=plt.axis()[:2], bins=20)
plt.xlabel('ref_mag')
plt.ylabel(f'{flux_type}_mag - ref_mag')
plt.title(f'v{visit}-{band}')

fig.add_subplot(2, 2, 3)
T = (df['base_SdssShape_xx'] + df['base_SdssShape_yy'])*0.2**2
plt.hexbin(df['ref_mag'], T, mincnt=1)
scp.plot_binned_stats(df['ref_mag'], T, x_range=plt.axis()[:2], bins=20)
plt.xlabel('ref_mag')
plt.ylabel('T (arcsec**2)')
plt.title(f'v{visit}-{band}')

fig.add_subplot(2, 2, 4)
scp.plot_detection_efficiency(butler, visit, df, ref_cat, x_range=(12, 25))
plt.title(f'v{visit}-{band}')

plt.tight_layout()
plt.savefig(args.outfile)
