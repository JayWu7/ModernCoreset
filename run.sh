#!/bin/bash


#Load environment

module load gcc/8.4.0 anaconda
source activate /home/wux4/.conda/envs/rapidsai

#Path setting

base_directory=/scratch/work/wux4/thesis/ModernCoreset/data/
output_directory=/scratch/work/wux4/thesis/ModernCoreset/output/


#Set hyper-parameters

data_name=$1  #Should be stored in base_directory
coreset_size=$2
data_dimension=$3

cluster_size=5  #kmeans cluster size

#Get the files path
file_name=${data_name}.csv
data_path=${base_directory}${file_name}
coreset_path=${output_directory}${data_name}-coreset_v.csv
coreset_weights_path=${output_directory}${data_name}-coreset_w.csv

data_centers_path=${output_directory}${data_name}-centers.csv
data_labels_path=${output_directory}${data_name}-labels.csv
coreset_centers_path=${output_directory}${data_name}-coreset_v-centers.csv
coreset_labels_path=${output_directory}${data_name}-coreset_v-labels.csv

#Repeat the experiments for 5 times to get the average result
for loop in 1 2 3 4 5
do
echo "Experiment: ${loop} start!"
#Generate coreset
/scratch/work/wux4/thesis/ModernCoreset/cuda/main $data_path $coreset_size $cluster_size $data_dimension $output_directory || exit 1;
#Run cuml
python cuml_evaluate.py ${data_path} ${coreset_path} ${coreset_weights_path} $cluster_size || exit 1;
#Evaluate
/scratch/work/wux4/thesis/ModernCoreset/cuda/evaluate ${data_path} ${data_centers_path} ${data_labels_path} ${coreset_path} ${coreset_weights_path} ${coreset_centers_path} ${coreset_labels_path} || exit 1;        
done

