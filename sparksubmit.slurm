#!/bin/bash
#SBATCH --job-name=spark-pi      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=80000              # memory per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH -p cox                   # partition
#SBATCH -e /n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x128_spark/Code/connected-components-3d/sparkjob.err
#SBATCH -o /n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x128_spark/Code/connected-components-3d/sparkjob.out

module load Anaconda3/5.0.1-fasrc02

source activate fillholes

export PYTHONPATH=$PYTHONPATH:/n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x128_spark/Code/connected-components-3d/

cd /n/pfister_lab2/Lab/tfranzmeyer/Data/1024x1024x128_spark/Code/connected-components-3d/

spark-submit --num-executors 1 --executor-cores 1 --executor-memory 5gb runspark.py  
