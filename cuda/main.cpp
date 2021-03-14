//
// Created by Xiaobo Wu on 2021/2/26.
//

#include "kmeans.h"
#include "kmeans.cpp"
#include "dataloader.h"
#include "dataloader.cpp"
#include <iostream>

using namespace std;
using namespace coreset;


int main(){
    DataLoader<float> dataloader;
    vector<vector<float> > data = dataloader.Loader("hayes-roth.csv");
    //vector<vector<float> > sampled_data = dataloader.DataSample(data, 1000);  // sample 1000 items of data
    //cout<<data.size()<<endl;
    //cout<<data[0].size()<<endl;
    KMeans kmeans(8, "k-means++",10, 10);
    vector<vector<float> > init_centers = kmeans.KMeans_pp_Init(data, 5);
    
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






    return 0;
}


