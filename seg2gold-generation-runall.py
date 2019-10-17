import os

SLURM_OUTPUT_FOLDER = '/n/home12/bmatejek/neuronseg/COX_SLURMS'

rhoana_files = os.listdir('rhoana')

template = '''#!/bin/bash
#
# add all other SBATCH directives here
#
#SBATCH -p cox                                                         # use the COX partition
#SBATCH -n 1                                                           # Number of cores
#SBATCH -N 1                                                           # Ensure that all cores are on one matching
#SBATCH --mem=60000                                                    # CPU memory in GBs
#SBATCH -t 0-02:00                                                     # time in dd-hh:mm to run the code for
#SBATCH --mail-type=ALL                                                # send all email types (start, end, error, etc.)
#SBATCH --mail-user=bmatejek@g.harvard.edu                             # email address to send to
#SBATCH -o /n/home12/bmatejek/neuronseg/COX_SLURM_LOG/{JOBNAME}.out    # where to write the log files
#SBATCH -e /n/home12/bmatejek/neuronseg/COX_SLURM_LOG/{JOBNAME}.err    # where to write the error files

source new-modules.sh
module load Anaconda3
module load cuda/9.0-fasrc02 cudnn/7.1_cuda9.0-fasrc01

source activate ibex_env

export PYTHONPATH=$PYTHONPATH:/n/home12/bmatejek/

cd /n/home12/bmatejek/neuronseg/

python scripts/seg2gold-generation-runone.py {PREFIX}

echo "DONE"

'''

files_written = 0

for rhoana_file in rhoana_files:
    prefix = rhoana_file[:-3]

    output_filename = 'cache/{}-seg2gold.map'.format(prefix)
    if os.path.exists(output_filename): continue

    jobname = 'seg2gold-{}'.format(prefix)

    t = template
    t = t.replace('{JOBNAME}', jobname)
    t = t.replace('{PREFIX}', prefix)

    with open('{}/{}.slurm'.format(SLURM_OUTPUT_FOLDER, jobname), 'w') as fd:
        fd.write(t)

    files_written += 1

print 'Wrote {} slurms files to {}.'.format(files_written, SLURM_OUTPUT_FOLDER)
