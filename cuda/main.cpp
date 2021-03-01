//
// Created by Xiaobo Wu on 2021/2/26.
//

#include "kmeans.h"
#include "dataloader.h"

using namespace std;
using namespace coreset;


int main(){
    DataLoader<float> dataloader;
    vector<vector<float> > data = dataloader.Loader("Active Wiretap_dataset.csv");
    vector<vector<float> > sampled_data = dataloader.DataSample(data, 1000);  // sample 1000 items of data
    
    KMeans kmeans;
    vector<vector<float> > init_centers = kmeans.KMeans_pp_Init(sampled_data, 5);
    cout<<"End!"<<endl;

    return 0;
}


