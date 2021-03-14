//
// Created by Xiaobo Wu on 2021/1/18.
//

#ifndef MODERNCORESET_CUDA_POINTS_H
#define MODERNCORESET_CUDA_POINTS_H
typedef unsigned long long int size_int;

#include <vector>

using namespace std;
namespace coreset{
    class Points {
    private:
        size_int size;
        unsigned int dimension;
        vector<vector<float> > values;
        vector<float> weights;

    public:
        Points();

        Points(size_int size, unsigned int dimension);

        Points(size_int size, unsigned int dimension, vector<vector<float> > values);

        Points(size_int size, unsigned int dimension, vector<vector<float> > values, vector<float> weights);

	Points(size_int size, unsigned int dimension, float* values, float* weights);

        unsigned long int Size();

        void FillPoints(vector<vector<float> > values, vector<float> weights = vector<float>());

        void AddPoints(vector<vector<float> > values, vector<float> weights = vector<float>());

        void SetWeights(vector<float> weights);

        vector<vector<float> > GetValues();

        vector<float> GetWeights();

        unsigned int Dimension();

    };
}

#endif //MODERNCORESET_CUDA_POINTS_H
