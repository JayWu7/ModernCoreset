//
// Created by Xiaobo Wu on 2021/1/25.
//
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
#include <string>
#include "points.h"


#ifndef MODERNCORESET_CUDA_KMEANS_H
#define MODERNCORESET_CUDA_KMEANS_H

typedef unsigned long long int size_int;

using namespace std;

namespace coreset {
    class KMeans {
    private:
        //vector<vector<float>> points;

        long double cost;   // objective cost

        vector<vector<float>> centers;

        int n_clusters, max_iter;

        string init; // support types: ['k-means++', 'random']

        vector<int> labels;

    protected:
        float Square(float value);
        float Squared_L2_Distance(vector<float> &first, vector<float> &second);
        void VectorAddition(vector<float> &first, vector<float> &second);
        void VectorAddition(vector<float> &output, vector<float> &first, vector<float> &second);
        void VectorDivide(vector<float> &point, int  n);

    public:
        KMeans(int n_clusters = 8, string init = "k-means++", int n_init=10, int max_iter=300);

        void Fit(vector<vector<float>> &points, const vector<float> &weights=vector<float>());

        long double GetCost();

        vector<int> GetLabel();

        vector<vector<float>> GetCenters();

        vector<vector<float>> KMeans_pp_Init(vector<vector<float>> &points, int n_cluster = 8);

        vector<vector<float>> KMeans_rd_Init(vector<vector<float>> &points, int n_cluster = 8);

    };

}

#endif //MODERNCORESET_CUDA_KMEANS_H
