#!/usr/bin/env python
import argparse
import desc.sims_ci_pipe as scp

parser = argparse.ArgumentParser()
parser.add_argument('instcat', help='Instance catalog')
parser.add_argument('--detectors', nargs='+', default=None, type=int,
                    help='Sensors for which to generate grid of stars, e.g., '
                    '`--detectors 94 95 96`.'
                    'If omitted, then simulate all sensors')
args = parser.parse_args()

grid_instcat = scp.make_star_grid_instcat(args.instcat,
                                          detectors=args.detectors)
scp.make_reference_catalog(grid_instcat)
