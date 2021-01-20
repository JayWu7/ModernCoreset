//
// Created by Xiaobo Wu on 2021/1/18.
//

#ifndef MODERNCORESET_CUDA_POINTS_H
#define MODERNCORESET_CUDA_POINTS_H

#include <vector>

using namespace std;
namespace coreset{
    class Points {
    private:
        unsigned long int size;
        unsigned int dimension;
        vector<vector<float>> values;
        vector<float> weights;

    public:
        Points();

        Points(unsigned long int size, unsigned int dimension);

        Points(unsigned long int size, unsigned int dimension, vector<vector<float>> values);

        Points(unsigned long int size, unsigned int dimension, vector<vector<float>> values, vector<float> weights);

        unsigned long int Size();

        void FillPoints(vector<vector<float>> values, vector<float> weights = vector<float>());

        void AddPoints(vector<vector<float>> values, vector<float> weights = vector<float>());

        void SetWeights(vector<float> weights);

        vector<vector<float>> GetValues();

        vector<float> GetWeights();

        unsigned int Dimension();

    };
}

#endif //MODERNCORESET_CUDA_POINTS_H
