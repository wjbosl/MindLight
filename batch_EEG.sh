#!/bin/bash
# Sample batchscript to run a job array for different EEG files with one batchscript

#SBATCH --partition=bch-compute # queue to be used
#SBATCH --time=2:00:00 # Running time (in hours-minutes-seconds)
#SBATCH --job-name=epi-compute #PLA Job name
#SBATCH --mail-type=BEGIN,END,FAIL # send and email when the job begins, ends or fails
#SBATCH --mail-user=william.bosl@childrens.harvard.edu # Email address to send the job status
#SBATCH --output=/lab-share/CHIP-Bosl-e2/Public/Data/ISP/All/Output/isp_%A_%a.txt        # Name of the output file
#SBATCH --nodes=1 # Number of compute nodes
#SBATCH --ntasks=1 # Number of cpu cores on one node
#SBATCH --mem=1000M
#SBATCH --array=1-6                                                                           # number of processors

source /programs/biogrids.shrc

#export search_dir="/lab-share/CHIP-Bosl-e2/Public/Data/Supriya/Eyes_closed/C01"
#export search_dir="/lab-share/CHIP-Bosl-e2/Public/Data/Epilepsy/Absence/EDF"                   # input directory
#export search_dir="/lab-share/CHIP-Bosl-e2/Public/Data/ISP/All"                   # input directory
export search_dir="/home/ch116278/Public/Data/Maski/IED"
#export search_dir="/lab-share/CHIP-Bosl-e2/Public/Data/PLAAY"
#export search_dir="/lab-share/CHIP-Bosl-e2/Public/Data/Cook"
#FILES=($search_dir/BP_c71c3039.edf)    
#FILES=($search_dir/*.edf)                                                                               # filenames
#FILES=($search_dir/*.mat)    
FILES=($search_dir/IED*.EDF)    
N=$(echo ${#FILES[@]})

# Get all the filenames in the relevant directory
FILE=${FILES[$SLURM_ARRAY_TASK_ID - 1 ]}
#OUTPUTFILE="cook_1999Hz_$SLURM_ARRAY_TASK_ID.csv"
#OUTPUTFILE="Results/bects_$SLURM_ARRAY_TASK_ID.csv"
#OUTPUTFILE="$FILE_$SLURM_ARRAY_TASK_ID.csv"
#OUTPUTFILE="Supriya/C01-2_$SLURM_ARRAY_TASK_ID.csv"
OUTPUTFILE="/lab-share/CHIP-Bosl-e2/Public/Data/Maski/IED/Output/IED_$FILE_$SLURM_ARRAY_TASK_ID.csv"   # output filenames

echo "job array is starting with file $FILE at `date` files"

#touch test-"$SLURM_ARRAY_TASK_ID".txt
#> Med1/"$SLURM_ARRAY_TASK_ID".txt
#echo "This is test number $SLURM_ARRAY_TASK_ID" >> test-"$SLURM_ARRAY_TASK_ID".txt
echo "Processing file $FILE " >> test-"$SLURM_ARRAY_TASK_ID".txt

# Run the python processing on each file
python.pyrqa process_single_EEG.py $FILE $OUTPUTFILE input_parameters.txt                               # Input parameters
 
echo "job array is finished at `date`"

### Save
#######SBATCH --output=cook_1999Hz_%A_%a.txt # Name of the output file
