/*
Merge and Reduce version of Coreset method. 
Utilize the key property: 'Composable' to paralize the computation of coreset, 
and deal with the situation that the input data is ouersize the storage of GPU memory.
*/

#include "coreset.cu"


coreset::FlatPoints
compute_coreset_mr(vector<float> &points, vector<float> &data_weights, 
    unsigned int dimension, unsigned int n_cluster, size_int n_coreset, unsigned int chunk_size){
    
    size_int n = points.size() / dimension;
    
    if(n <= chunk_size){
	coreset::FlatPoints data_chunk(n, dimension);
	data_chunk.FillPoints(points, data_weights);
        return data_chunk;
    }
    
    size_int half_size = n / 2;
    size_int half_size_point = half_size * dimension;
    
    vector<float> left_part_points(points.begin(), points.begin() + half_size_point);
    vector<float> right_part_points(points.begin() + half_size_point, points.end());
    vector<float> left_part_weights(data_weights.begin(), data_weights.begin() + half_size);
    vector<float> right_part_weights(data_weights.begin() + half_size, data_weights.end());
    
    // Free the memories of vector points and data_weights
    vector<float>().swap(points);
    vector<float>().swap(data_weights);

    //size_int left_n_coreset = n_coreset / 2;
    //size_int right_n_coreset = n_coreset - left_n_coreset;

    coreset::FlatPoints left_part;
    coreset::FlatPoints right_part;
    
    // Reduce the question to left and right part sub-questions
    left_part = compute_coreset_mr(left_part_points, left_part_weights, dimension, n_cluster, n_coreset, chunk_size);
    right_part = compute_coreset_mr(right_part_points, right_part_weights, dimension, n_cluster, n_coreset, chunk_size);
    // Merge the left and right part coreset
    coreset::FlatPoints coreset; //define coreset object
    coreset.AddPoints(dimension, left_part.GetValues(), left_part.GetWeights());
    coreset.AddPoints(dimension, right_part.GetValues(), right_part.GetWeights());
    vector <float> values = coreset.GetValues();
    vector <float> weights = coreset.GetWeights();
    // Compute and return the coreset of merged previous layer coreset
    return compute_coreset_flat(values, weights, dimension, n_cluster, n_coreset);
}

