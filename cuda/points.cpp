//
// Created by Xiaobo Wu on 2021/1/18.
//

#include "points.h"
using namespace std;

namespace coreset {

    Points::Points() {
        this->size = 0;
        this->dimension = 0;
        this->values = vector<vector<float> >();
        this->weights = vector<float>();
    }

    Points::Points(size_int size, unsigned int dimension) {
        this->size = size;
        this->dimension = dimension;
        this->values = vector<vector<float> >(size, vector<float>(dimension));
        this->weights = vector<float>(size);
    }

    Points::Points(size_int size, unsigned int dimension, vector<vector<float> > values) {
        this->size = size;
        this->dimension = dimension;
        if (this->size != values.size())
            throw "values size is not matched setting size!";
        this->values = values;
        this->weights = vector<float>(size);
    }
    
    Points::Points(size_int size, unsigned int dimension, vector<vector<float> > values, vector<float> weights) {
        this->size = size;
        this->dimension = dimension;
        if (this->size != values.size())
            throw "values size is not matched setting size!";
        this->values = values;
        if (this->size != weights.size())
            throw "Weights size is not matched setting size!";
        this->weights = weights;
    }
    
    /*
    Points::Points(size_int size, unsigned int dimension, float* values, float* weights) {
        this->size = size;
        this->dimension = dimension;
	this->values = vector<vector<float> >(size, vector<float>(dimension));
        this->weights = vector<float>(weights, weights + size);
	
	//transfor the data from arrays to vector form
	for(int i=0; i<size; i++){
	    size_int start_id = i * dimension;
	    for(int j=0; j<dimension; j++){
	        this.values[i][j] = values[start_id + j];
	    }
	}
        
    }*/


    unsigned long int Points::Size() {
        return this->size;
    }

    unsigned int Points::Dimension() {
        return this->dimension;
    }

    void Points::FillPoints(vector<vector<float> > values, vector<float> weights) {
        if (this->size != values.size() && this->dimension != values[0].size())
            throw ("Please input the values with the shape of %d,%d.", this->size, this->dimension);
        this->values = values;
        if (weights.empty()) {
            this->weights = vector<float>(this->size, 1.0);
        } else if (this->size != weights.size()) {
            throw ("Please input the weights with the length of %d.", this->size);
        }
        else{
            this->weights = weights;
        }
    }
    
    void Points::AddPoints(vector<vector<float> > values, vector<float> weights) {
	if (this->values.empty())  //Right now, it's empty in the points object
	    this->dimension = values[0].size();

        if (this->dimension != values[0].size())
            throw ("Please add the points with same dimension %d as current points", this->dimension);

        this->values.insert(this->values.end(), values.begin(), values.end());
        if (weights.empty()){
            this->weights.insert(this->weights.end(), values.size(), 1.0);
        }
        else if (values.size() != weights.size()){
            throw "The new values and weights are not in the same length.";
        }
        else{
            this->weights.insert(this->weights.end(), weights.begin(), weights.end());
        }

        this->size += values.size();
    }

    void Points::SetWeights(vector<float> weights) {
        if (weights.size() != this->size){
            throw "The new weights are not in the same length with the values.";
        }
        this->weights = weights;
    }

    vector<vector<float> > Points::GetValues() {
        return this->values;
    }

    vector<float> Points::GetWeights() {
        return this->weights;
    }

}









