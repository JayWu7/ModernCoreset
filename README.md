# ModernCoreset

We implement a coreset library that can construct a small coreset for arbitrary shapes of numerical data with a decent time cost. The implementation was mainly based on the coreset construction algorithm that proposed by [Braverman et al. (SODA 2021)](https://epubs.siam.org/doi/pdf/10.1137/1.9781611976465.159). The experiments were conducted to benchmark with the [CuML](https://github.com/rapidsai/cuml) library, a well-known collection of the efficient GPU implementation of various machine learning algorithms. The benchmark results prove our implementation of the coreset method can lead to a fast speed in solving the k-means problem among the big data with high accuracy in the meantime. In addition, by utilizing the composable property of the coreset, our merge-and-reduce version of the coreset breaks through the restriction of GPU memory which is a heavy issue in CuML k-means method.



## Results

#### Accuracy versus coreset size

##### a. Accuracy evaluation method:

In the accuracy evaluation, first, we use our coreset method to get the small coreset of the big original data. Then we utilize the k-means function provided in CuML on top of coreset to get k clustering centers. On the other hand, we also get k clustering centers among the original big data by the same method k-means in CuML. Finally, we compute the sum of objective values in the original data with both two sets of centers separately. Set the two objective values sums are $obj_c$ (objective values for coreset) and $obj_o$ (objective values for original data set), then we define the relative error of the coreset method:
$$
error = \frac{obj_c - obj_o}{obj_o}
$$
Note that we do not use the absolute value here, which means the error can be negative. **And suppose the error is a negative value; in this case, it means the clustering result generated upon the coreset is even better than the result from the original big data**.

We mainly conduct the experiments in the same data set with different coreset sizes to test the influence of coreset size on its accuracy. The values of coreset size that we used in this experiment are: **100, 200, 500, 800, 1000, 2000, 5000, 8000, 10000, 20000, 50000, 80000, 100000**.

##### b. Data sets:

1. [**Heterogeneity Activity Recognition**](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition) data Set, this is a data set from the smart electrical devices like smart photos and smartwatch that aims to record the statistics of humans’ activities in the real-world context. There are four similar files in this data set, and we select the **Watch_gyroscope.csv** file in our experiment. In more detail, this file includes 3205431 points with five attributes in each point, which refers to have the dimension of ***(3205431, 5)***.

2. [**US Census Data (1990)**](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)) data Set, this is a high dimensional data for classification and clustering problem. In total, it has 68 different numeric attributes, and it has 2458285 points. Therefore the dimension of this data set is ***(2458285, 68)***. The principal reason for choosing this data set is to utilize its high dimension property to test the coreset performance in the high dimensional data set.
3. [**Open Street Map**](https://en.wikipedia.org/wiki/OpenStreetMap) , which is a public repository that aims to generate and distribute accessible geographic data for the whole world. Basically, it supplies detailed position information, including the longitude and latitude of the places around the world.  We utilize osm data for Hong Kong, China in this experiment, which has the dimension for ***(2366214, 2)***. 



##### c. Coreset accuracy:

For the computation of the relative error of the coreset, we repeat the same experiments five times in total, and we compute the average relative error of the coreset as the final error. We test the coreset accuracy with three different data set with relatively low, medium and high dimension that mentioned above, which refers to the dimension of 2, 5 and 68, respectively. Note that we set the clustering size to 5 in our experiments. We plot the results as a diagram, see it below:

![Coreset Relative Error Changes with the Coreset Size in data set Watch gyroscope, USCensus1990, and hk osm](/Users/jaywu/Desktop/thesis/project/ModernCoreset/results/plot_1.png)



#### Speed of generation versus data size

##### a. Data sets:

We utilize the latest Open Street Map geographic data of European countries in the experiments of this evaluation. More specificly, we download and extract the osm location data for nine countries, including Finland, Sweden, Denmark, Norway, France, Germany, Netherlands, British, Poland. Then we combined this nine osm file to a single big CSV file that is in the shape of **(1156442555, 2)**. For convenience, we named this big CSV file as **all-latest.csv**. 

##### b. Coreset generation speed:

 We test the coreset generation speed along with the change of input data size. This experiment is conducted in **all-latest.csv** file, which has more than one billion points. Furthermore, the sampling strategy is used here to get the different size of sub-sets in the same file. We select the test input data size to: $ \mathbf{10^4, 5 \cdot 10^4, 10^5, 5 \cdot 10^5, 10^6, 5 \cdot 10^6, 10^7, 5 \cdot 10^7, 10^8, 5 \cdot 10^8, 10^9}$. The coreset size is fixed to 8000 since we find that the coreset size to 8000 will usually produce the nearly best result in the previous accuracy experiments. In addition, we set the clustering size to 5 just as in previous experiments. And we conduct this experiment by randomly samples different size of sub-sets in **all-latest.csv** file. The diagram of the coreset construction speed along with the growth of input data size is located below:

![Coreset Construction Speed versus the size of input data](/Users/jaywu/Desktop/thesis/project/ModernCoreset/results/plot_2.png)

#### Merge-and-Reduce 

The merge-and-reduce version results in oversized data (data size is over than the GPU memory) is the most significant result of our project since the main motivation for us to carry on this project is to clarify whether the coreset method can exceed the GPU memory size limitation when dealing with big data. In this experiment, we set the coreset size to N = 1000, clustering size to k = 5.

Our experiment results show that when CuML facing the super colossal data set, it will raise a **"cudaErrorMemoryAllocation out of memory"** error. And it is worth noting that we found CuML even can not deal with the data size of around **5.6G** in CSV form with a **16G** GPU memory. This 5.6G CSV file is the latest osm data file for Germany, which in shape of ***(333418873, 2)***.

As a contrast, our merge-and-reduce coreset implementation can handle this size level data set. It cost **101.01** seconds to construct the coreset in the Germany osm data file that we described above. Furthermore, we test the merge-and-reduce implementation in **all_latest.csv** data set, with shape ***(1156442555, 2)***,  The result shows the merge-and-reduce version coreset method can successfully process this data set and generate a coreset in only a few minutes (**393.82** seconds, to be precise). And just as we presented above, our implementation can deal with any size of data since it read data in a stream and invoke the coreset computation chunk by chunk.

## Environment

#### GPU and Cuda environment:

1. Tesla P100 GPU card with Pascal architecture and 16 GB memory in our experiments.

2. cuda/11.0.2
3. using [Aalto Triton cluster](https://scicomp.aalto.fi/triton/).

#### CuML environment:

1. CuML (0.17.0)
2. see CuML installing instructions at: https://rapids.ai/start.html#rapids-release-selector

#### Python environment:

```
pip install -r requirements.txt
```

#### 

## Usage

##### Python implementation

We build a naive Python single-thread version coreset computation method to prove that our method work.

###### A simple usage of constructing coreset from csv file:

```
cd python   #get into python implementation directory
```

```python
import numpy as np
from points import Points
from dataloader import loader
from coreset import compute_coreset

data = loader(filename='hayes-roth.csv')
centers, ids = initial_cluster(data, 5)
coreset = compute_coreset(data, 5, 50)
print(centers)
print(ids)
print(coreset.get_values())
print(coreset.get_weights())
```



###### Comparing the objective values of k-means between our Coreset method and CuML:

```python
import sys
from cuda_kmeans import cuml_kmeans_csv

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Parameters error!')
        print("Usage: 'python cuml_evaluate.py <original_data_path> <coreset_path> <coreset_weights_path> <cluster_size>'")
        exit() 
    data_path = sys.argv[1]
    coreset_path = sys.argv[2]
    coreset_weights_path = sys.argv[3]
    cluster_size = int(sys.argv[4])
    
    sample_size = None
    if len(sys.argv) > 5: #optional parameters
        sample_size = int(sys.argv[5])
        

    cuml_kmeans_csv(data_path, cluster_size, sample_size=sample_size)
    cuml_kmeans_csv(coreset_path, cluster_size, csv_weights_path=coreset_weights_path)
```

```bash
python cuml_evaluate.py <data_path> <coreset_path> <coreset_weights_path> <cluster_size>
```



##### C++/Cuda based implementation

we use C++ and Cuda to implement the multi-threads parallel version coreset method; and finally we use CUDA to implement the merge-and-reduce version of coreset construction method.

###### Compile:

```bash
nvcc -o main main.cu       #compile creset method
nvcc -o mr_main mr_main.cu      #compile merge-and-reduce coreset method
```

We utilize the basic implementation of coreset construction method in merge-and-reduce version.

###### Using basic gpu version of coreset method:

```bash
#!/bin/bash

#Load environment/ In Aalto Triton server

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
python python/cuml_evaluate.py ${data_path} ${coreset_path} ${coreset_weights_path} $cluster_size || exit 1;
#Evaluate
/scratch/work/wux4/thesis/ModernCoreset/cuda/evaluate ${data_path} ${data_centers_path} ${data_labels_path} ${coreset_path} ${coreset_weights_path} ${coreset_centers_path} ${coreset_labels_path} || exit 1;        
done
```

###### Using merge-and-reduce version to deal with any size-level input data:

```bash
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
chunk_size=5000000

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
for loop in 1 2
do
echo "Experiment: ${loop} start!"
#Generate coreset
/scratch/work/wux4/thesis/ModernCoreset/cuda/mr_main $data_path $coreset_size $cluster_size $data_dimension $output_directory $chunk_size || exit 1;
#Run cuml
#python python/cuml_evaluate.py ${data_path} ${coreset_path} ${coreset_weights_path} $cluster_size || exit 1;
#Evaluate
#/scratch/work/wux4/thesis/ModernCoreset/cuda/evaluate ${data_path} ${data_centers_path} ${data_labels_path} ${coreset_path} ${coreset_weights_path} ${coreset_centers_path} ${coreset_labels_path} || exit 1;        
done
```

#### Contact

xiaobo.wu@aalto.fi, or jaywu16@163.com

