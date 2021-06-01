#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=5-00 #runs for 5 days max
#SBATCH --mem=100G

#repeat the experiments with different coreset_size seeting: (1,2,5,10)*(100,1000,10000)
filename=denmark-latest
dimension=2
for size in 10000 50000 100000 500000 1000000 5000000 10000000 50000000 100000000 500000000 1000000000
do
echo "*************************************************" 
echo "Current size is: $size"
filename=all_$size
bash ./mr_run.sh $filename 8000 $dimension

done

