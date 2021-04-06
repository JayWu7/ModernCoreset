//
// Created by Xiaobo Wu on 2021/2/26.
//

#include "kmeans.h"
#include "kmeans.cpp"
#include "dataloader.h"
#include "dataloader.cpp"
#include "coreset.cu"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>


using namespace std;
using namespace coreset;


int main(){
    DataLoader<float> dataloader(6); //dimension = 6
    vector<float> data = dataloader.Loader_1D("hayes-roth.csv");
    //vector<vector<float> > sampled_data = dataloader.DataSample(data, 1000);  // sample 1000 items of data
    
    /*KMeans kmeans(8, "k-means++",10, 10);
    vector<vector<float> > init_centers = kmeans.KMeans_pp_Init(data, 5);
    for(int i=0; i<init_centers.size(); i++){
        vector<float> center = init_centers[i];
	for(int j=0; j<center.size(); j++)
	    cout<<center[j]<<" ";
	cout<<endl;
    }*/

    //Test Kmeans method:
    //kmeans.Fit(sampled_data);
    // cout<<kmeans.GetCenters().size()<<endl;
    // cout<<kmeans.GetCost()<<endl;
    // vector<int> labels=kmeans.GetLabel();
    // for(int i=0; i<labels.size(); i++)
    // {
    //     cout<<labels[i]<<endl;
    // }


    //Test Coreset method:
    //thrust::device_vector<float> device_points(data.begin(), data.end());
    unsigned int dimension = dataloader.dimension;
    unsigned int n_cluster = 5;
    size_int n = data.size() / dimension; 
    //float centers[n_cluster * dimension];
    float points[data.size()];
    copy(data.begin(), data.end(), points);
    //k_means_pp_init_cu(points, n, centers, n_cluster, dimension);
    
    unsigned int n_coreset = 20;
    coreset::Points coreset(n_coreset, dimension);

    compute_coreset(coreset, data, dimension, n_cluster, n_coreset);

    return 0;
}

