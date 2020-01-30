import os
import param
import numpy as np
import sys

template = '''#!/bin/bash
#
# add all other SBATCH directives here
#
#SBATCH -p {PARTITION}                                       # use the COX partition
#SBATCH -n 1                                                 # Number of cores
#SBATCH -N 1                                                 # Ensure that all cores are on one matching
#SBATCH --mem={MEMORY}                                       # CPU memory in MBs
#SBATCH -t 0-{HOURS}:00                                            # time in dd-hh:mm to run the code for
#SBATCH --mail-type=NONE                                   # send all email types (start, end, error, etc.)
#SBATCH --mail-user=tfranzmeyer@g.harvard.edu                # email address to send to
#SBATCH -o {OUTPUT_PATH}/{JOBNAME}.out                       # where to write the log files
#SBATCH -e {ERROR_PATH}/{JOBNAME}.err                        # where to write the error files
#SBATCH -J fillholes{JOBNAME}                               # jobname given to job



source activate fillholes

export PYTHONPATH=$PYTHONPATH:{RUNCODEDIRECTORY}

cd {RUNCODEDIRECTORY}

python {COMMAND}

echo "DONE"

'''


def makeFolder(folder_path):
    if os.path.exists(folder_path):
        raise ValueError("Folderpath " + folder_path + " already exists!")
    else:
        os.mkdir(folder_path)

def writeFile(filename, data):
    if os.path.exists(filename):
        raise ValueError("File " + filename + " already exists!")
    else:
        with open(filename, 'w') as fd:
            fd.write(data)


if(len(sys.argv))!=5:
    raise ValueError(" Scripts needs 4 cluster partitions as input, put 0 if not less desired")
else:
    n_part = 0
    partitions = ["0","0","0","0"]

    if sys.argv[1]!="0":
        partitions[0] = sys.argv[1]
        n_part +=1
    if sys.argv[2]!="0":
        partitions[1] = sys.argv[2]
        n_part +=1
    if sys.argv[3]!="0":
        partitions[2] = sys.argv[3]
        n_part +=1
    if sys.argv[4]!="0":
        partitions[3] = sys.argv[4]
        n_part +=1

files_written = 0

template = template.replace('{RUNCODEDIRECTORY}', param.code_path)
template = template.replace('{HOURS}', param.run_hours)


SLURM_OUTPUT_FOLDER = param.slurm_path

step01folderpath = SLURM_OUTPUT_FOLDER+"step01/"
step02Afolderpath = SLURM_OUTPUT_FOLDER+"step02A/"
step02Bfolderpath = SLURM_OUTPUT_FOLDER+"step02B/"
step03folderpath = SLURM_OUTPUT_FOLDER+"step03/"

makeFolder(step01folderpath)
makeFolder(step02Afolderpath)
makeFolder(step02Bfolderpath)
makeFolder(step03folderpath)

# Write Slurm for preparations file
command = "preparation.py"
jobname = "step00"

# write slurm for step one
for bz in range(param.z_start, param.z_start + param.n_blocks_z):
    for by in range(param.y_start, param.y_start + param.n_blocks_y):

        command = "stepOne.py" + " " + str(bz) + " " + str(by)
        jobname = "S01"+"_" +"z"+str(bz).zfill(2)+"y"+str(by).zfill(2)

        t = template
        t = t.replace('{JOBNAME}', jobname)
        t = t.replace('{COMMAND}', command)
        t = t.replace('{ERROR_PATH}', param.error_path)
        t = t.replace('{OUTPUT_PATH}', param.output_path)
        t = t.replace('{MEMORY}', param.memory_step01)
        t = t.replace('{PARTITION}', partitions[np.random.randint(0,n_part)])
        filename = step01folderpath + jobname + ".slurm"
        writeFile(filename, t)
        files_written += 1

# write slurm for step two A
for bz in range(param.z_start, param.z_start + param.n_blocks_z):
    for by in range(param.y_start, param.y_start + param.n_blocks_y):

        command = "stepTwoA.py" + " " + str(bz) + " " + str(by)
        jobname = "S2A_" +"z"+str(bz).zfill(2)+"y"+str(by).zfill(2)

        t = template
        t = t.replace('{JOBNAME}', jobname)
        t = t.replace('{COMMAND}', command)
        t = t.replace('{ERROR_PATH}', param.error_path)
        t = t.replace('{OUTPUT_PATH}', param.output_path)
        t = t.replace('{MEMORY}', param.memory_step02A)
        t = t.replace('{PARTITION}', partitions[np.random.randint(0,n_part)])

        filename = step02Afolderpath + jobname + ".slurm"
        writeFile(filename, t)
        files_written += 1

# Write Slurm for step two B
command = "stepTwoB.py"
jobname = "step02B"

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)
t = t.replace('{ERROR_PATH}', param.error_path)
t = t.replace('{OUTPUT_PATH}', param.output_path)
t = t.replace('{MEMORY}', param.memory_step02B)
t = t.replace('{PARTITION}', partitions[np.random.randint(0,n_part)])

filename = step02Bfolderpath + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

# write slurm for step three
for bz in range(param.z_start, param.z_start + param.n_blocks_z):
    for by in range(param.y_start, param.y_start + param.n_blocks_y):

        command = "stepThree.py" + " " + str(bz) + " " + str(by)
        jobname = "S3"+"_"+ "z"+str(bz).zfill(2)+"y"+str(by).zfill(2)

        t = template
        t = t.replace('{JOBNAME}', jobname)
        t = t.replace('{COMMAND}', command)
        t = t.replace('{ERROR_PATH}', param.error_path)
        t = t.replace('{OUTPUT_PATH}', param.output_path)
        t = t.replace('{MEMORY}', param.memory_step03)
        t = t.replace('{PARTITION}', partitions[np.random.randint(0,n_part)])

        filename = step03folderpath + jobname + ".slurm"
        writeFile(filename, t)
        files_written += 1


print ("Files written: " + str(files_written))
