//
// Created by Xiaobo Wu on 2021/1/25.
//

#include "kmeans.h"

#define FLOAT_MAX_VALUE 3.40282e+038

using namespace std;

namespace coreset {

    KMeans::KMeans(int n_clusters, string init, int n_init, int max_iter) {
        this->init = init;
        this->max_iter = max_iter;
        this->n_clusters = n_clusters;
    }

    float KMeans::Square(float value) {
        return value * value;
    }

    float KMeans::Squared_L2_Distance(vector<float> &first, vector<float> &second) {
        float distance = 0.0;
        int dimension = first.size();
        if (dimension != second.size()) {
            throw "Input two vector are not in the same dimension!";
        }
        for (int i = 0; i < dimension; i++) {
            distance += pow(first[i] - second[i], 2);
        }
        return distance;
    }

    void KMeans::VectorAddition(vector<float> &first, vector<float> &second) {
        int dimension = first.size();
        if (dimension != second.size()) {
            throw "Input two vector are not in the same dimension!";
        }
        for (int i = 0; i < dimension; i++) {
            first[i] += second[i];
        }
    }


    void KMeans::VectorAddition(vector<float> &output, vector<float> &first, vector<float> &second) {
        int dimension = first.size();
        if (dimension != second.size()) {
            throw "Input two vector are not in the same dimension!";
        }
        for (int i = 0; i < dimension; i++) {
            output[i] = first[i] + second[i];
        }
    }

    void KMeans::VectorDivide(vector<float> &point, int n) {
        int dimension = point.size();
        for (int i = 0; i < dimension; i++) {
            point[i] /= n;
        }
    }


    long double KMeans::GetCost() {
        return this->cost;
    }

    vector<vector<float> > KMeans::GetCenters() {
        return this->centers;
    }

    vector<int> KMeans::GetLabel() {
        return this->labels;
    }

    void KMeans::Fit(vector<vector<float> > &points, const vector<float> &weights) {
        /*
         * Core function of KMeans method.
         * */
        size_int size = points.size();
        int dimension = points[0].size();
        vector<int> assigns(size);  // clusters assignments of each point
        vector<vector<float> > centers;

        if (this->init == "k-means++"){
            centers = this->KMeans_pp_Init(points, this->n_clusters);
        }
        else if(this->init == "random"){
            centers = this->KMeans_rd_Init(points, this->n_clusters);
        }
        else{
            throw "Only support ['k-means++', 'random'] at this moment.";
        }

        float cur_min_dis; // Current minimum distance
        float cur_dis;
        int cur_assigned_center;
        vector<vector<float> > new_centers(n_clusters, vector<float>(dimension, 0.0));
        vector<int> count(n_clusters, 0);
        long double cost;

        for (int iteration = 0; iteration < this->max_iter; iteration++) {
            cost = 0.0;
            for (size_int i = 0; i < points.size(); i++){
                cur_min_dis = FLOAT_MAX_VALUE;
                for (int j = 0; j < this->n_clusters; j++){
                    cur_dis = this->Squared_L2_Distance(points[i], centers[j]);
                    if (cur_dis < cur_min_dis){
                        cur_min_dis = cur_dis;
                        cur_assigned_center = j; // The index of the cluster
                    }
                }
                cost += cur_min_dis;
                assigns[i] = cur_assigned_center;
                count[cur_assigned_center] += 1;
                VectorAddition(new_centers[cur_assigned_center], points[i]);
            }
            // Update process
            for (int k = 0; k < n_clusters; k++){
                VectorDivide(new_centers[k], count[k]);
                centers[k] = new_centers[k];
            }

            this->centers = centers;
            this->labels = assigns;
            this->cost = cost;

//            if (){
//                // end the iteration since small changes among process
//                break;
//            }
        }
    }

    vector<vector<float> > KMeans::KMeans_pp_Init(vector<vector<float> > &points, int n_cluster) {
        /*
         * KMeans++ initialization
        */
        if (n_cluster < 1) {
            throw "n_cluster, the number of clusters should at least greater than 1.";
        }
        vector<vector<float> > centers;
        size_int size = points.size();
        vector<float> dist(size, FLOAT_MAX_VALUE);
        vector<float> weights(size);
        long double dist_sum;

        //Choose one center uniformly at random among the data points.
        random_device rd;  //Will be used to obtain a seed for the random number engine
        mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        uniform_int_distribution<> distrib(0, size - 1);
        int first_center_index = distrib(gen);
        centers.push_back(points[first_center_index]);  // Add the first center into the centers list

        int cur_center_index;
        for (int i = 0; i < n_cluster - 1; i++) {
            dist_sum = 0.0;
            for (size_int j = 0; j < size; j++) {
                float new_dis = this->Squared_L2_Distance(centers[i], points[j]);
                if (new_dis < dist[j]) {
                    dist[j] = new_dis;
                }
                dist_sum += dist[j];
            }
            for (size_int j = 0; j < size; j++)
                weights[j] = dist[j] / dist_sum;

            discrete_distribution<> dd(weights.begin(), weights.end());
            cur_center_index = dd(gen);
            centers.push_back(points[cur_center_index]);
        }

        return centers;
    }

    vector<vector<float> > KMeans::KMeans_rd_Init(vector<vector<float> > &points, int n_cluster) {
        /*
         * Random initialization of clusters
        */
        size_int size = points.size();
        random_device rd;  //Will be used to obtain a seed for the random number engine
        mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        vector<int> indices(size);
        for (size_int i = 0; i < size; i++)
            indices[i] = i;
        shuffle(indices.begin(), indices.end(), gen);
        uniform_int_distribution<> distrib(0, size - 1);
        int start_index = distrib(gen);
        vector<vector<float> > centers(n_cluster);
        int index;
        for (int i = 0; i < n_cluster; i++) {
            index = (start_index + i) % size;
            centers[i] = points[indices[index]];
        }
        return centers;
    }

}