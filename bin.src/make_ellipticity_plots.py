#!/usr/bin/env python
import argparse
import desc.sims_ci_pipe as scp

parser = argparse.ArgumentParser()
parser.add_argument('repo', help='Data repo with calexps')
parser.add_argument('--outfile', type=str, default=None,
                    help='Name of png file to contain plots')
parser.add_argument('--opsim_db', type=str, default=None,
                    help='opsim db to use for getting visit info')
args = parser.parse_args()

scp.ellipticity_distributions(args.repo, outfile=args.outfile,
                              opsim_db=args.opsim_db)
