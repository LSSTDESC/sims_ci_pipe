#!/usr/bin/env python
import sys
import yaml
from desc.sims_ci_pipe import pipeline_stages

config_file = sys.argv[1]
with open(config_file) as fd:
    pipeline = yaml.safe_load(fd)['stages']

for stage_name in pipeline:
    stage = pipeline_stages[stage_name](config_file)
    print(stage.stage_name)
    stage.run()
