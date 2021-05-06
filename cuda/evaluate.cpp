#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include "dataloader.h"
#include "dataloader.cpp"
#include <chrono>

using namespace std;
using namespace coreset;


long double kmeans_inertia(vector<vector<float> > &points, vector<vector<float> > &centers, vector<unsigned int> &labels, vector<float> weights=vector<float>()){
	long double cost = 0.0;
	if(weights.empty()){
		for (int i=0; i<points.size(); i++){
			long double squa_dis = 0.0;
			for (int j=0; j<points[0].size(); j++){
				squa_dis += pow(points[i][j] - centers[labels[i]][j], 2);
			}
			cost += squa_dis;
		}	
	}
	else{
		for (int i=0; i<points.size(); i++){
                        long double squa_dis = 0.0;
                        for (int j=0; j<points[0].size(); j++){
                                squa_dis += (weights[i] * pow(points[i][j] - centers[labels[i]][j], 2));
                        }
                        cost += squa_dis;
                }
	}

	return cost;
}


//Evaluate the kmeans objective value in both coreset and original set with different weights 
void evaluation_1(string coreset_path, string coreset_centers_path, string coreset_labels_path, string coreset_weights_path, string data_path, string data_centers_path, string data_labels_path, string data_weights_path=""){

	using timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
	auto print_exec_time = [](timepoint start, timepoint stop) {
		auto duration_us = chrono::duration_cast<chrono::microseconds>(stop - start);
		auto duration_ms = chrono::duration_cast<chrono::milliseconds>(stop - start);
    		auto duration_s = chrono::duration_cast<chrono::seconds>(stop - start);

	    	cout << duration_us.count() << " us | " << duration_ms.count() << " ms | "
        		      << duration_s.count() << " s\n";
		  };

	auto start = std::chrono::high_resolution_clock::now(); //Starting time	
	DataLoader<float> dataloader;
	DataLoader<unsigned int> labelsloader;

	vector<vector<float> > coreset = dataloader.ReadCsv(coreset_path);
	vector<vector<float> > coreset_centers = dataloader.ReadCsv(coreset_centers_path);
	vector<unsigned int> coreset_labels = labelsloader.ReadCsv_1D(coreset_labels_path);
	vector<float> coreset_weights = dataloader.ReadCsv_1D(coreset_weights_path);
	vector<vector<float> > data = dataloader.ReadCsv(data_path);
	vector<vector<float> > data_centers = dataloader.ReadCsv(data_centers_path);
	vector<unsigned int> data_labels = labelsloader.ReadCsv_1D(data_labels_path);
	vector<float> data_weights;
	if(data_weights_path != "")
		data_weights = dataloader.ReadCsv_1D(data_weights_path);	
	else
		data_weights = vector<float>();
	
	long double coreset_inertia = kmeans_inertia(coreset, coreset_centers, coreset_labels, coreset_weights);
	long double original_data_inertia = kmeans_inertia(data, data_centers, data_labels, data_weights);	
        
	float relative_error = abs(coreset_inertia - original_data_inertia) / original_data_inertia;
        
	auto stop = std::chrono::high_resolution_clock::now();
        cout << "Execution Time: ";
        print_exec_time(start, stop);
        cout<<"Coreset cost: " << coreset_inertia << endl;
	cout<<"Original data cost: " << original_data_inertia << endl;
        cout<<"Relative Error: " << relative_error << endl;
}


int main(int argc, char **argv){

        if (argc != 8) {
                cout << "Parameters errors!"<<endl;
   		cout << "Usage: ./evaluate <original_data_path)> <original_data_centers_path> <original_data_labels_path> <coreset_path> <coreset_weights_path> <coreset_centers_path> <coreset_labels_path>\n";
		return EXIT_FAILURE;
    	}
        string data_path = argv[1];
        string data_centers_path = argv[2];
	string data_labels_path = argv[3];
 	string coreset_path = argv[4];
	string coreset_weights_path = argv[5];
	string coreset_centers_path = argv[6];
	string coreset_labels_path = argv[7];

	evaluation_1(coreset_path, coreset_centers_path, coreset_labels_path, coreset_weights_path, data_path, data_centers_path, data_labels_path);	
	return 0;
}

