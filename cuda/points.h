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

	void FillPoints(vector<float> values, vector<float> weights = vector<float>());

        void AddPoints(vector<vector<float> > values, vector<float> weights = vector<float>());

        void SetWeights(vector<float> weights);

        vector<vector<float> > GetValues();

        vector<float> GetWeights();

        unsigned int Dimension();

    };


    class FlatPoints {
    	private:
		size_int size;
       		unsigned int dimension;
       		vector<float> values;
	        vector<float> weights;
   	 public:
		FlatPoints();
		FlatPoints(size_int size, unsigned int dimension);
		unsigned long int Size();
		unsigned int Dimension();
		vector<float>  GetValues();
		vector<float> GetWeights();
		void FillPoints(vector<float> values, vector<float> weights = vector<float>());
		void AddPoints(unsigned int dimension, vector<float> values, vector<float> weights = vector<float>() );
    };

}

#endif //MODERNCORESET_CUDA_POINTS_H
