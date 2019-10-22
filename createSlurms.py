import os
import param

template = '''#!/bin/bash
#
# add all other SBATCH directives here
#
#SBATCH -p shared                                            # use the COX partition
#SBATCH -n 1                                                 # Number of cores
#SBATCH -N 1                                                 # Ensure that all cores are on one matching
#SBATCH --mem={MEMORY}                                       # CPU memory in MBs
#SBATCH -t 0-01:00                                           # time in dd-hh:mm to run the code for
#SBATCH --mail-type=NONE                                     # send all email types (start, end, error, etc.)
#SBATCH --mail-user=tfranzmeyer@g.harvard.edu                # email address to send to
#SBATCH -o {OUTPUT_PATH}/{JOBNAME}.out                       # where to write the log files
#SBATCH -e {ERROR_PATH}/{JOBNAME}.err                        # where to write the error files
#SBATCH -J fillholes_{JOBNAME}                               # jobname given to job

module load Anaconda3/5.0.1-fasrc02
module load cuda/9.0-fasrc02 cudnn/7.1_cuda9.0-fasrc01

source activate fillholes

export PYTHONPATH=$PYTHONPATH:/n/home12/tfranzmeyer/

cd /n/home12/tfranzmeyer/Code/connected-components-3d/

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

files_written = 0

memory_std = param.memory_needed
memory_step04 = memory_std*param.n_blocks_z*param.n_blocks_y*param.n_blocks_x

SLURM_OUTPUT_FOLDER = '/n/home12/tfranzmeyer/slurm_files/'

step00folderpath = SLURM_OUTPUT_FOLDER+"step00/"
step01folderpath = SLURM_OUTPUT_FOLDER+"step01/"
step02Afolderpath = SLURM_OUTPUT_FOLDER+"step02A/"
step02Bfolderpath = SLURM_OUTPUT_FOLDER+"step02B/"
step02Bfolderpath = SLURM_OUTPUT_FOLDER+"step02C/"
step03folderpath = SLURM_OUTPUT_FOLDER+"step03/"
step04folderpath = SLURM_OUTPUT_FOLDER+"step04/"

makeFolder(step00folderpath)
print(step00folderpath)
makeFolder(step01folderpath)
makeFolder(step02Afolderpath)
makeFolder(step02Bfolderpath)
makeFolder(step02Cfolderpath)
makeFolder(step03folderpath)
makeFolder(step04folderpath)

# Write Slurm for preparations file
command = "preparation.py"
jobname = "step00"+"_"+param.outp_ID

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)
t = t.replace('{ERROR_PATH}', param.error_path_preparation)
t = t.replace('{OUTPUT_PATH}', param.output_path_preparation)
t = t.replace('{MEMORY}', "2000")

filename = step00folderpath + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

# write slurm for step one
for bz in range(param.z_start, param.z_start + param.n_blocks_z):
    for by in range(param.y_start, param.y_start + param.n_blocks_y):
        for bx in range(param.x_start, param.x_start + param.n_blocks_x):

            command = "stepOne.py" + " " + str(bz) + " " + str(by) + " " + str(bx)
            jobname = "step01"+"_"+param.outp_ID+"_" +"z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

            t = template
            t = t.replace('{JOBNAME}', jobname)
            t = t.replace('{COMMAND}', command)
            t = t.replace('{ERROR_PATH}', param.error_path)
            t = t.replace('{OUTPUT_PATH}', param.output_path)
            t = t.replace('{MEMORY}', str(memory_std))

            filename = step01folderpath + jobname + ".slurm"
            writeFile(filename, t)
            files_written += 1

# write slurm for step two A
for bz in range(param.z_start, param.z_start + param.n_blocks_z):

    command = "stepTwoA.py" + " " + str(bz)
    jobname = "step02A"+"_"+param.outp_ID+"_" +"z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

    t = template
    t = t.replace('{JOBNAME}', jobname)
    t = t.replace('{COMMAND}', command)
    t = t.replace('{ERROR_PATH}', param.error_path)
    t = t.replace('{OUTPUT_PATH}', param.output_path)
    t = t.replace('{MEMORY}', str(memory_std))

    filename = step02Afolderpath + jobname + ".slurm"
    writeFile(filename, t)
    files_written += 1

# write slurm for step two B
for it in range(2, param.iterations_needed+1)
    for bz in range(param.z_start, param.z_start + param.n_blocks_z):

        command = "stepTwoB.py" + " " + str(bz) + " " + str(it)
        jobname = "step02A"+"_"+param.outp_ID+"_" +"z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

        t = template
        t = t.replace('{JOBNAME}', jobname)
        t = t.replace('{COMMAND}', command)
        t = t.replace('{ERROR_PATH}', param.error_path)
        t = t.replace('{OUTPUT_PATH}', param.output_path)
        t = t.replace('{MEMORY}', str(memory_std))

        filename = step02Bfolderpath + jobname + ".slurm"
        writeFile(filename, t)
        files_written += 1


# Write Slurm for step two C
command = "stepTwoC.py"
jobname = "step02"+"_"+param.outp_ID

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)
t = t.replace('{ERROR_PATH}', param.error_path)
t = t.replace('{OUTPUT_PATH}', param.output_path)
t = t.replace('{MEMORY}', str(memory_std))

filename = step02Cfolderpath + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

# write slurm for step three
for bz in range(param.z_start, param.z_start + param.n_blocks_z):
    for by in range(param.y_start, param.y_start + param.n_blocks_y):
        for bx in range(param.x_start, param.x_start + param.n_blocks_x):

            command = "stepThree.py" + " " + str(bz) + " " + str(by) + " " + str(bx)
            jobname = "step03"+"_"+param.outp_ID+"_"+ "z"+str(bz).zfill(4)+"y"+str(by).zfill(4)+"x"+str(bx).zfill(4)

            t = template
            t = t.replace('{JOBNAME}', jobname)
            t = t.replace('{COMMAND}', command)
            t = t.replace('{ERROR_PATH}', param.error_path)
            t = t.replace('{OUTPUT_PATH}', param.output_path)
            t = t.replace('{MEMORY}', str(memory_std))

            filename = step03folderpath + jobname + ".slurm"
            writeFile(filename, t)
            files_written += 1

# Write Slurm for step four
command = "stepFour.py"
jobname = "step04"+"_"+param.outp_ID

t = template
t = t.replace('{JOBNAME}', jobname)
t = t.replace('{COMMAND}', command)
t = t.replace('{ERROR_PATH}', param.error_path)
t = t.replace('{OUTPUT_PATH}', param.output_path)
t = t.replace('{MEMORY}', str(memory_step04))

filename = step04folderpath + jobname + ".slurm"
writeFile(filename, t)
files_written += 1

print ("Files written: " + str(files_written))
