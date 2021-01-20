//
// Created by Xiaobo Wu on 2021/1/19.
//

#include "dataloader.h"

using namespace std;

namespace coreset {

    template<class T>
    DataLoader<T>::DataLoader() {
        // todo
    }


    template<class T>
    string DataLoader<T>::PathJoin(string path, string file, char sep) {
        if (path[path.size() - 1] != sep) {
            path += sep;
        }

        return path + file;
    }

    template<class T>
    void DataLoader<T>::ExistedCheck(string filename) {
        //todo
    }

    template<class T>
    vector<vector<T>> DataLoader<T>::DataFilter(vector<vector<T> > data) {
        // todo
    }

    template<class T>
    vector<vector<T>> DataLoader<T>::DataSample(vector<vector<T> > data, unsigned long int size) {
        if (size > data.size())
            size = data.size();

        vector<vector<T>> samples(data.begin(), data.begin() + size);
        return samples;
    }

    template<class T>
    vector<vector<T>> DataLoader<T>::ReadCsv(string filepath, char sep) {
        // todo
        vector<vector<T>> data;
        ifstream fp(filepath);
        string line;
        getline(fp, line); // skip the first line
        while (getline(fp, line)) {
            vector<T> data_line;
            string value = "";
            for (int i = 0; i < line.size(); i++){
                if (line[i] == sep){
                    data_line.push_back(stringToNum(value));
                    value = "";
                }
                else{
                    value += line[i];
                }
            }
            data_line.push_back(stringToNum(value)); // add the last number
            data.push_back(data_line); // add to data vector
        }
        return data;
    }


    template<class T>
    vector<vector<T>> DataLoader<T>::Loader(string filename, string sep, string file_type) {
        //ExistedCheck(filename);
        string file_path = PathJoin("./data", filename);
        vector<vector<T>> data;
        if (file_type == "csv") {
            data = ReadCsv(file_path);
        } else if (file_type == "txt") {
            //todo
        } else {
            throw ("We don't support s% file type at this moment.", file_type);
        }

        //data = DataFilter(data);
        return data;
    }

    template<class T>
    T DataLoader<T>::stringToNum(const string &str) {
        istringstream iss(str);
        T num;
        iss >> num;
        return num;
    }
}

