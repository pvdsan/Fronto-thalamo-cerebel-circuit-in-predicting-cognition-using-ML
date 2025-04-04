#!/bin/bash
#SBATCH -p qTRDHM
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8                                                  
#SBATCH --mem=50G
#SBATCH -e error%A.err 
#SBATCH -o out%A.out
#SBATCH -A trends395c149
#SBATCH -J Working_Memory_Prediction
#SBATCH --oversubscribe
#SBATCH --mail-user=pvdsan@gmail.com


# a small delay at the start often helps
sleep 2s 

#activate the environment
source /home/users/sdeshpande8/anaconda3/bin/activate CognitionPrediction

# CD into your directory
cd /data/users4/sdeshpande8/Classical_Methods_ROI_Mean_Data/

# run the matlab batch script
python SVR_Perm.py

# a delay at the end is also good practice
sleep 10s