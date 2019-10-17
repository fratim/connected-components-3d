import os

SLURM_OUTPUT_FOLDER = '/n/home12/tfranzmeyer/slurm_files/'

template = '''#!/bin/bash
#
# add all other SBATCH directives here
#
#SBATCH -p serial_requeue                                                                                                                           # use the COX partition
#SBATCH -n 1                                                                                                                                        # Number of cores
#SBATCH -N 1                                                                                                                                        # Ensure that all cores are on one matching
#SBATCH --mem=10000                                                                                                                                 # CPU memory in GBs
#SBATCH -t 0-00:10                                                                                                                                  # time in dd-hh:mm to run the code for
#SBATCH --mail-type=end                                                                                                                             # send all email types (start, end, error, etc.)
#SBATCH --mail-user=tfranzmeyer@g.harvard.edu                                                                                                       # email address to send to
#SBATCH -o /n/home12/tfranzmeyer/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/ZF_concat_6to7_0512_0512/output_files/{JOBNAME}.out       # where to write the log files
#SBATCH -e /n/home12/tfranzmeyer/wiring/raw_data/segmentations/Zebrafinch/stacked_volumes/ZF_concat_6to7_0512_0512/error_files/{JOBNAME}.err        # where to write the error files

source new-modules.sh
module load Anaconda/5.0.1-fasrc01
module load cuda/9.0-fasrc02 cudnn/7.1_cuda9.0-fasrc01

source activate holefilling

export PYTHONPATH=$PYTHONPATH:/n/home12/tfranzmeyer/

cd /n/home12/tfranzmeyer/connected-components-3d/

python stepOne.py {PREFIX}

echo "DONE"

'''

files_written = 0

bx = 0
by = 0

for bz in range(4):
    # prefix = rhoana_file[:-3]
    #
    # output_filename = 'cache/{}-seg2gold.map'.format(prefix)
    # if os.path.exists(output_filename): continue

    prefix = str(bz) + " " + str(by) + " " + str(bx)

    # jobname = 'seg2gold-{}'.format(prefix)
    jobname = "z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

    t = template
    t = t.replace('{JOBNAME}', jobname)
    t = t.replace('{PREFIX}', prefix)

    with open(SLURM_OUTPUT_FOLDER + jobname + ".slurm", 'w') as fd:
        fd.write(t)

    files_written += 1

print ("Files written: " + str(files_written))
