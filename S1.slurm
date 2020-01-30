#!/bin/bash
#
# add all other SBATCH directives here
#
#SBATCH -p remotedesktop                                       # use the COX partition
#SBATCH -n 1                                                 # Number of cores
#SBATCH -N 1                                                 # Ensure that all cores are on one matching
#SBATCH --mem=35000                                       # CPU memory in MBs
#SBATCH -t 0-3:00                                      # time in dd-hh:mm to run the code for
#SBATCH --mail-type=NONE                                     # send all email types (start, end, error, etc.)
#SBATCH --mail-user=tfranzmeyer@g.harvard.edu                # email address to send to
#SBATCH -o /n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x1024/thinning/output_files//S1.out                       # where to write the log files
#SBATCH -e /n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x1024/thinning/error_files//S1.err                        # where to write the error files
#SBATCH -J fillholes_S1                               # jobname given to job

module load Anaconda3/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.1_cuda9.0-fasrc01

source activate fillholes

export PYTHONPATH=$PYTHONPATH:/n/home12/tfranzmeyer/Code/

cd /n/home12/tfranzmeyer/Code/skeletons/examples/

python scripts/step1.py

echo "DONE"

