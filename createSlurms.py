import os
import param
from functions import makeFolder

template = '''#!/bin/bash
#
# add all other SBATCH directives here
#
#SBATCH -p serial_requeue                                                                                                                           # use the COX partition
#SBATCH -n 1                                                                                                                                        # Number of cores
#SBATCH -N 1                                                                                                                                        # Ensure that all cores are on one matching
#SBATCH --mem=10000                                                                                                                                 # CPU memory in GBs
#SBATCH -t 0-00:10                                                                                                                                  # time in dd-hh:mm to run the code for
#SBATCH --mail-type=NONE                                                                                                                             # send all email types (start, end, error, etc.)
#SBATCH --mail-user=tfranzmeyer@g.harvard.edu                                                                                                       # email address to send to
#SBATCH -o {ERROR_PATH}/{JOBNAME}.out       # where to write the log files
#SBATCH -e {OUTPUT_PATH}/{JOBNAME}.err        # where to write the error files

module load Anaconda3/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.1_cuda9.0-fasrc01

source activate fillholes

export PYTHONPATH=$PYTHONPATH:/n/home12/tfranzmeyer/

cd /n/home12/tfranzmeyer/Code/connected-components-3d/

python {COMMAND}

echo "DONE"

'''

def writeFile(filename, data):
    if os.path.exists(filename):
        raise ValueError("File " + filename + " already exists!")
    else:
        with open(filename, 'w') as fd:
            fd.write(data)

files_written = 0

SLURM_OUTPUT_FOLDER = '/n/home12/tfranzmeyer/slurm_files/'

error_path = param.error_path
output_path = param.output_path

step00folderpath = SLURM_OUTPUT_FOLDER+"step00/"
step01folderpath = SLURM_OUTPUT_FOLDER+"step01/"
step02folderpath = SLURM_OUTPUT_FOLDER+"step02/"
step03folderpath = SLURM_OUTPUT_FOLDER+"step03/"
step04folderpath = SLURM_OUTPUT_FOLDER+"step04/"

makeFolder(step00folderpath)
print(step00folderpath)
makeFolder(step01folderpath)
makeFolder(step02folderpath)
makeFolder(step03folderpath)
makeFolder(step04folderpath)

# Write Slurm for preparations file
command = "preparation.py"
jobname = "step00"

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)
t = t.replace('{ERROR_PATH}', error_path)
t = t.replace('{OUTPUT_PATH}', output_path)

filename = step00folderpath + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

# write slurm for step one

for bz in range(param.n_blocks_z):
    for by in range(param.n_blocks_y):
        for bx in range(param.n_blocks_x):

            command = "stepOne.py" + " " + str(bz) + " " + str(by) + " " + str(bx)
            jobname = "step01_" +"z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

            t = template
            t = t.replace('{JOBNAME}', jobname)
            t = t.replace('{COMMAND}', command)
            t = t.replace('{ERROR_PATH}', error_path)
            t = t.replace('{OUTPUT_PATH}', output_path)

            filename = step01folderpath + jobname + ".slurm"
            writeFile(filename, t)
            files_written += 1

# Write Slurm for step two
command = "stepTwo.py"
jobname = "step02"

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)
t = t.replace('{ERROR_PATH}', error_path)
t = t.replace('{OUTPUT_PATH}', output_path)

filename = step02folderpath + jobname + ".slurm"
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
            t = t.replace('{ERROR_PATH}', error_path)
            t = t.replace('{OUTPUT_PATH}', output_path)

            filename = step03folderpath + jobname + ".slurm"
            writeFile(filename, t)
            files_written += 1

# Write Slurm for step four
command = "stepFour.py"
jobname = "step04"

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)
t = t.replace('{ERROR_PATH}', error_path)
t = t.replace('{OUTPUT_PATH}', output_path)

filename = step04folderpath + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

print ("Files written: " + str(files_written))
