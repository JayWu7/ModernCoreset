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
#include <iostream>

using namespace std;

namespace coreset {
template <class T>
    class DataLoader {
    //private:

    public:
        DataLoader();

        vector<vector<T> > Loader(string filename, char sep = ',', string file_type = "csv");

        void ExistedCheck(string filename);

        vector<vector<T> > ReadCsv(string filepath, char sep = ',');

        string PathJoin(string path, string file, char sep = '/');

        vector<vector<T> > DataFilter(vector<vector<T> > data);

        vector<vector<T> > DataSample(vector<vector<T> > data, unsigned long int size);

        T stringToNum(const string &str); // string to numerical value
    };

}

#endif //MODERNCORESET_CUDA_DATALOADER_H
