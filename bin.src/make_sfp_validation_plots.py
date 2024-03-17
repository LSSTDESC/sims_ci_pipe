#!/usr/bin/env python
import argparse
import desc.sims_ci_pipe as scp

parser = argparse.ArgumentParser()
parser.add_argument('repo', help='Data repo with calexps')
parser.add_argument('visit', type=int, help='visit to process')
parser.add_argument('--outdir', type=str, default='.',
                    help='directory to contain output files')
parser.add_argument('--flux_type', type=str, default='base_PsfFlux',
                    help='flux type of point sources to use')
parser.add_argument('--opsim_db', type=str, default=None,
                    help='opsim db to use for getting visit info')
parser.add_argument('--instrument', type=str, default='LSSTCam-imSim',
                    help='instrument to use')
parser.add_argument('--collection', type=str, default=None,
                    help='collection to use')
parser.add_argument('--detector', type=int, default=None,
                    help='detector to use')
args = parser.parse_args()

if args.collection is not None:
    collections = [args.collection]
else:
    collections = None
scp.sfp_validation_plots(args.repo, args.visit, args.outdir,
                         flux_type=args.flux_type, opsim_db=args.opsim_db,
                         instrument=args.instrument, collections=collections,
                         detector=args.detector)
