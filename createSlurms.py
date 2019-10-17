import os
import param

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

module load Anaconda3/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.1_cuda9.0-fasrc01

source activate fillholes

export PYTHONPATH=$PYTHONPATH:/n/home12/tfranzmeyer/

cd /n/home12/tfranzmeyer/Code/connected-components-3d/

python {COMMAND}

echo "DONE"

'''

def writefile(filename, data):
    if os.path.exists(filename):
        raise ValueError("File " + filename + " already exists!")
    else:
        with open(filename, 'w') as fd:
            fd.write(data)

files_written = 0

# Write Slurm for preparations file
command = "preparations.py"
jobname = "step00"

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)

filename = SLURM_OUTPUT_FOLDER + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

# write slurm for step one

for bz in range(param.n_blocks_z):
    for by in range(param.n_blocks_y):
        for bx in range(param.n_blocks_x):

            command = "stepOne.py"str(bz) + " " + str(by) + " " + str(bx)
            jobname = "step01_" +"z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

            t = template
            t = t.replace('{JOBNAME}', jobname)
            t = t.replace('{COMMAND}', command)

            filename = SLURM_OUTPUT_FOLDER + jobname + ".slurm"
            writeFile(filename, t)
            files_written += 1

# Write Slurm for step two
command = "stepTwo.py"
jobname = "step02"

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)

filename = SLURM_OUTPUT_FOLDER + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

# write slurm for step three
for bz in range(param.n_blocks_z):
    for by in range(param.n_blocks_y):
        for bx in range(param.n_blocks_x):

            command = "stepThree.py" + " " + str(bz) + " " + str(by) + " " + str(bx)
            jobname = "step03_" + "z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

            t = template
            t = t.replace('{JOBNAME}', jobname)
            t = t.replace('{COMMAND}', command)

            filename = SLURM_OUTPUT_FOLDER + jobname + ".slurm"
            writeFile(filename, t)
            files_written += 1

# Write Slurm for step four
command = "stepFour.py"
jobname = "step04"

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)

filename = SLURM_OUTPUT_FOLDER + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

print ("Files written: " + str(files_written))
