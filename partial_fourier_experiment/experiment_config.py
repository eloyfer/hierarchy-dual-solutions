import yaml
import os

project_dir = '/cs/labs/nati/eloyfer/projects/hierarchy-dual-solutions/partial_fourier_experiment/'
working_dir = os.path.join(project_dir, 'outputs')
slurm_dir = os.path.join(working_dir, 'slurm')
krawtchouks_dir = os.path.join(working_dir, 'saved_krawtchouks')
db_file = os.path.join(working_dir, 'experiment.db')
