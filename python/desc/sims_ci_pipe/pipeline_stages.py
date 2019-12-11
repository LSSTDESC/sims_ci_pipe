"""
Stages for continuous integration pipeline.
"""
import os
import glob
import shutil
from collections import defaultdict
import subprocess
import yaml
import pandas as pd
import lsst.daf.persistence as dp


__all__ = ['pipeline_stages', 'get_visits', 'merge_metric_files']


def get_visit_info(instcat):
    """
    Get the visit and band from an instance catalog.

    Parameters
    ----------
    instcat: str
        Instance catalog.

    Returns
    -------
    (int, str)
    """
    with open(instcat) as fd:
        for line in fd:
            if line.startswith('filter'):
                band = 'ugrizy'[int(line.strip().split()[1])]
            elif line.startswith('obshistid'):
                visit = int(line.strip().split()[1])
            if line.startswith('object'):
                break
    return visit, band


def get_visits(repo, dataset_type='raw'):
    """
    Get the visit and band info as a dictionary from a data repo using
    the butler.

    Parameters
    ----------
    repo: str
        The data repo containing the selected dataset_type.
    dataset_type: str ['raw']
        The dataset type to query for.

    Returns
    -------
    dict with visits as the keys and the corresponding bands as the values.
    """
    butler = dp.Butler(repo)
    datarefs = butler.subset(dataset_type)
    visits = dict()
    for dataref in datarefs:
        md = dataref.get(f'{dataset_type}_md')
        visits[dataref.dataId['visit']] = md.getScalar('FILTER')
    return visits


def merge_metric_files(metric_files):
    """
    Merge the metric files into a single dataframe.

    Parameters
    ----------
    metric_files: list
        List of pickled dataframe metric files.

    Returns
    -------
    pandas.DataFrame containing the merged data frames.
    """
    df = pd.read_pickle(metric_files[0])
    for item in metric_files[1:]:
        df = df.append(pd.read_pickle(item))
    return df


class PipelineStage:
    """Base class for sims_ci_pipe stages."""
    def __init__(self, config_file):
        """
        Parameters
        ----------
        config_file: str
            Yaml file containing the pipeline configuration.
        """
        with open(config_file) as fd:
            config = yaml.safe_load(fd)
        pipe_config = config['pipeline']
        self.config = config['stages']
        self.bands = pipe_config['bands']
        self.dry_run = pipe_config['dry_run']
        self.run_dir = os.path.join(os.path.abspath('.'),
                                    f'{pipe_config["run_number"]:05d}')
        os.makedirs(self.run_dir, exist_ok=True)
        shutil.copy(config_file,
                    os.path.join(self.run_dir, os.path.basename(config_file)))
        self.log_dir = os.path.join(self.run_dir, pipe_config['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)
        self.repo_dir = os.path.join(self.run_dir, pipe_config['repo_dir'])
        self.fits_dir = os.path.join(self.run_dir, pipe_config['fits_dir'])

    def execute(self, command, do_raise=False):
        """
        Use subprocess.check_call to execute a command line.

        Parameters
        ----------
        command: str
            The command line to run in a shell.
        do_raise: bool [False]
            Flag to re-raise any caught CalledProcessErrors.  If False,
            then ignore any such errors.
        """
        print(command)
        if self.dry_run:
            return
        try:
            subprocess.check_call(command, shell=True)
        except subprocess.CalledProcessError as eobj:
            if do_raise:
                raise eobj


class ImsimStage(PipelineStage):
    """
    Stage to run imsim.py for the specified instance catalogs and
    runtime configuration.
    """
    stage_name = 'imsim'
    def __init__(self, config_file):
        """
        Parameters
        ----------
        config_file: str
            Yaml file containing the pipeline configuration.
        """
        super(ImsimStage, self).__init__(config_file)

    def run(self):
        """Run method to execute the commands."""
        config = self.config[self.stage_name]
        psf = config['psf']
        sensors = config['sensors']
        processes = config['processes']
        instcat_dir = config['instcat_dir']
        if instcat_dir == 'default':
            instcat_dir = os.path.join(os.environ['SIMS_CI_PIPE_DIR'],
                                       'data', 'instcats')
        instcats = sorted(glob.glob(os.path.join(instcat_dir, 'phosim_*.txt')))
        visits = defaultdict(list)
        for instcat in instcats:
            visit, band = get_visit_info(instcat)
            if band not in self.bands:
                continue
            visits[band].append(visit)
            command = f'time imsim.py {instcat} --psf {psf} --sensors "{sensors}" --log_level DEBUG --outdir {self.fits_dir} --create_centroid_file --processes {processes} --seed {visit}'
            if config['disable_sensor_model']:
                command += ' --disable_sensor_model'
            log_file = os.path.join(self.log_dir, f'imsim_v{visit}-{band}.log')
            command = f'({command}) >& {log_file}'
            self.execute(command)


class IngestStage(PipelineStage):
    stage_name = 'ingest_images'
    def __init__(self, config_file):
        """
        Parameters
        ----------
        config_file: str
            Yaml file containing the pipeline configuration.
        """
        super(IngestStage, self).__init__(config_file)

    def run(self):
        """Run method to execute the commands."""
        os.makedirs(self.repo_dir, exist_ok=True)
        mapper_file = os.path.join(self.repo_dir, '_mapper')
        if not os.path.isfile(mapper_file):
            with open(mapper_file, 'w') as output:
                output.write('lsst.obs.lsst.imsim.ImsimMapper\n')

        config = self.config[self.stage_name]
        for target in ('CALIB', 'ref_cats', 'calibrations'):
            src = config[target]
            dest = os.path.join(self.repo_dir, target)
            if not os.path.islink(dest):
                os.symlink(src, dest)

        command = f'(time ingestImages.py {self.repo_dir} {self.fits_dir}/lsst_a*) >& {self.log_dir}/ingest_images.log'
        self.execute(command)

class ProcessCcdsStage(PipelineStage):
    stage_name = 'process_ccds'
    def __init__(self, config_file):
        """
        Parameters
        ----------
        config_file: str
            Yaml file containing the pipeline configuration.
        """
        super(ProcessCcdsStage, self).__init__(config_file)

    def run(self):
        """Run method to execute the commands."""
        config = self.config[self.stage_name]
        processes = config['processes']
        visits = get_visits(self.repo_dir)
        print(visits)
        for visit, band in visits.items():
            if band not in self.bands:
                continue
            command = f'(time processCcd.py {self.repo_dir} --output {self.repo_dir} --id visit={visit} --processes {processes} --no-versions) >& {self.log_dir}/processCcd_{visit}.log'
            self.execute(command)


class SfpValidationStage(PipelineStage):
    stage_name = 'sfp_validation'
    def __init__(self, config_file):
        """
        Parameters
        ----------
        config_file: str
            Yaml file containing the pipeline configuration.
        """
        super(SfpValidationStage, self).__init__(config_file)

    def run(self):
        """Run method to execute the commands."""
        config = self.config[self.stage_name]
        opsim_db = config['opsim_db']
        outdir = os.path.join(self.run_dir, config['out_dir'])
        os.makedirs(outdir, exist_ok=True)
        visits = get_visits(self.repo_dir)
        for visit, band in visits.items():
            if band not in self.bands:
                continue
            outfile \
                = os.path.join(outdir, f'sfp_validation_v{visit}-{band}.png')
            pickle_file \
                = os.path.join(outdir, f'sfp_validation_v{visit}-{band}.pkl')
            metrics_file \
                = os.path.join(outdir, f'sfp_metrics_v{visit}-{band}.pkl')
            command = f'(time make_sfp_validation_plots.py {self.repo_dir} {visit} --opsim_db {opsim_db} --outfile {outfile} --pickle_file {pickle_file} --metrics_file {metrics_file}) >& {self.log_dir}/sfp_validation_v{visit}-{band}.log'
            self.execute(command)
        if self.dry_run:
            return
        metric_files = sorted(glob.glob(os.path.join(outdir, '*metrics*.pkl')))
        df = merge_metric_files(metric_files)
        df.to_pickle(os.path.join(outdir, 'sfp_metrics.pkl'))


pipeline_stages = {_.stage_name: _ for _ in
                   (ImsimStage, IngestStage, ProcessCcdsStage,
                    SfpValidationStage)}
