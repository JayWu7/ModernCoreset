//
// Created by Xiaobo Wu on 2021/2/26.
//

#include "kmeans.h"
#include "kmeans.cpp"
#include "dataloader.h"
#include "dataloader.cpp"
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
    thrust::device_vector<float> device_points(data.begin(), data.end());

    //for (int i=0; i<a.size(); i++){
      //  cout<<a[i][0]<<endl;
    //}
    //thrust::device_vector <thrust::device_vector<float> > device_points;
    //thrust::copy(host_points.begin(), host_points.end(), device_points.begin());
    //cout<<device_points.size()<<endl;
    
    return 0;
}


