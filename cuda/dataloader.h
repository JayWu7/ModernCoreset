//
// Created by Xiaobo Wu on 2021/1/19.
//

#ifndef MODERNCORESET_CUDA_DATALOADER_H
#define MODERNCORESET_CUDA_DATALOADER_H

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <random>

using namespace std;

namespace coreset {
template <class T>
    class DataLoader {
    //private:
    public:
        unsigned int dimension;

        DataLoader();

        DataLoader(unsigned int dimension);

        vector<vector<T> > Loader(string filename, char sep = ',', string file_type = "csv");

        vector<T> Loader_1D(string filename, char sep = ',', string file_type = "csv");

        void ExistedCheck(string filename);

        vector<vector<T> > ReadCsv(string filepath, char sep = ',');

        void WriteCsv(string out_path, vector<vector<T> > points);
	
	void NormCsv_1D(vector<T> &data);
	void NormCsv(vector<vector<T> > &data);
	void NormCsv(string filepath, char sep = ',');

        vector<T> ReadCsv_1D(string filepath, char sep = ',');  // Read data into 1-d format

        void WriteCsv_1D(string out_path, vector<T> points, unsigned int dimension = 1); // Write the 1-d format data

        string PathJoin(string path, string file, char sep = '/');

        vector<vector<T> > DataFilter(vector<vector<T> > data);

        vector<vector<T> > DataSample(vector<vector<T> > data, unsigned long int size);

        T stringToNum(const string &str); // string to numerical value
    };

}

#endif //MODERNCORESET_CUDA_DATALOADER_H
