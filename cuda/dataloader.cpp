//
// Created by Xiaobo Wu on 2021/1/19.
//

#include "dataloader.h"
#include <iostream>
#include <algorithm>
#include <limits>


using namespace std;

namespace coreset {

    template<class T>
    DataLoader<T>::DataLoader() {
        this->dimension = 0;
        cout << "Dataloader object is being created" << endl;
    }

    template<class T>
    DataLoader<T>::DataLoader(unsigned dimension) {
        this->dimension = dimension;
        cout << "Dataloader object is being created" << endl;
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

    // template<class T>
    // vector<vector<T> > DataLoader<T>::DataFilter(vector<vector<T> > data) {
    //     // todo
    //     return vector<vector<T> > test();
    // }

    template<class T>
    vector<vector<T> > DataLoader<T>::DataSample(vector<vector<T> > data, unsigned long int size) {
        if (size > data.size())
            size = data.size();
        random_device rd;  //Will be used to obtain a seed for the random number engine
        mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        shuffle(data.begin(), data.end(), gen); // shuffle the data
        vector<vector<T> > samples(data.begin(), data.begin() + size);
        return samples;
    }

    template<class T>
    vector<vector<T> > DataLoader<T>::ReadCsv(string filepath, char sep) {
        vector<vector<T> > data;
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
        fp.close();
        return data;
    }
    
    //Normalize data
    template<class T>   
    void DataLoader<T>::NormCsv(string filepath, char sep){
	vector<vector<T> > data = this->ReadCsv(filepath);

        unsigned long int le = data.size();
        unsigned int dimension = data[0].size();
	float float_max = numeric_limits<float>::max();
	float float_min = numeric_limits<float>::min();

        vector<float> min_value(dimension, float_max);
        vector<float> max_value(dimension, float_min);

        for(int i = 0; i < le; i++){
                for(int j = 0; j < dimension; j++){
                        if(data[i][j] > max_value[j]){
                                max_value[j] = data[i][j];
                        }else if(data[i][j] < min_value[j]){
                                min_value[j] = data[i][j];
                        }
                }
        }

        for(int i = 0; i < le; i++){
                for(int j = 0; j < dimension; j++){
                       data[i][j] = (data[i][j] - min_value[j]) / (max_value[j] - min_value[j]);
                }
        }
	
	this->WriteCsv(filepath, data);
    }

    template<class T>
    void DataLoader<T>::WriteCsv(string filepath, vector<vector<T> > points) {
        ofstream fp(filepath);
        size_t length = points.size();
        unsigned int dimension = points[0].size();
        for(int j=0; j<dimension - 1; j++){
		fp << j <<',';
	}
	fp << dimension - 1 << '\n';
        for (int i=0; i<length; i++){
            for (int j=0; j<dimension - 1; j++)
                fp << points[i][j] <<',';
            
            fp << points[i][dimension - 1] << '\n';
        }
    }

    template<class T>
    void DataLoader<T>::WriteCsv_1D(string filepath, vector<T> points, unsigned int dimension) {
        ofstream fp(filepath);
        size_t length = points.size() / dimension;
        for(int j=0; j<dimension - 1; j++){
                fp << j <<',';
        }
        fp << dimension - 1 << '\n';
        for (int i=0; i<length; i++){
            size_t start_index = i * dimension;
            for (int j=0; j<dimension - 1; j++)
                fp << points[start_index + j] <<',';
            fp << points[start_index + dimension - 1] << '\n';
        }
    }


    template<class T>
    vector<vector<T> > DataLoader<T>::Loader(string file_path, char sep, string file_type) {
        //ExistedCheck(filename);
        vector<vector<T> > data;
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
    vector<T> DataLoader<T>::ReadCsv_1D(string filepath, char sep) {
        vector<T> data;
        ifstream fp(filepath);
        string line;
        getline(fp, line); // skip the first line
        while (getline(fp, line)) {
            string value = "";
            for (int i = 0; i < line.size(); i++){
                if (line[i] == sep){
                    data.push_back(stringToNum(value));
                    value = "";
                }
                else{
                    value += line[i];
                }
            }
            data.push_back(stringToNum(value)); // add the last number
        }
        if (this->dimension !=0 && data.size() % this->dimension != 0){
            throw "Dataset error! Probably somewhere have missing";
        }
	fp.close();
        return data;
    }

    template<class T>
    void DataLoader<T>::NormCsv_1D(vector<T> &data) {
	float min_value = *min_element(data.begin(), data.end());
        float max_value = *max_element(data.begin(), data.end());
	
	unsigned long int le = data.size();
	for(int i = 0; i < le; i++){
		data[i] = (data[i] - min_value) / (max_value - min_value);
	}
    }


    template<class T>
    void DataLoader<T>::NormCsv(vector<vector<T> > &data){
        unsigned long int le = data.size();
	unsigned int dimension = data[0].size();
	
	float min_value = data[0][0];
	float max_value = data[0][0];
	
	for(int i = 0; i < le; i++){
		for(int j = 0; j < dimension; j++){
			if(data[i][j] > max_value){
				max_value = data[i][j];
			}else if(data[i][j] < min_value){
				min_value = data[i][j];
			}	
		}
	}	

        for(int i = 0; i < le; i++){
		for(int j = 0; j < dimension; j++){
 	               data[i][j] = (data[i][j] - min_value) / (max_value - min_value);
		}
        }
    }


    template<class T>
    vector<T>  DataLoader<T>::Loader_1D(string file_path, char sep, string file_type) {
        vector<T> data;
        if (file_type == "csv") {
            data = ReadCsv_1D(file_path);
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

