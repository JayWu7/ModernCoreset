/*
Merge and Reduce version of Coreset method. 
Utilize the key property: 'Composable' to paralize the computation of coreset, 
and deal with the situation that the input data is ouersize the storage of GPU memory.
*/

#include "coreset.cu"


coreset::Points
compute_coreset_mr(vector<float> &points, vector<float> &data_weights, 
    unsigned int dimension, unsigned int n_cluster, size_int n_coreset, unsigned int chunk_size){
    
    size_int n = points.size() / dimension;
    
    if(n <= chunk_size)
        return compute_coreset(points, data_weights, dimension, n_cluster, n_coreset);
    
    size_int half_size = n / 2;
    size_int half_size_point = half_size * dimension;
    
    vector<float> left_part_points(points.begin(), points.begin() + half_size_point);
    vector<float> right_part_points(points.begin() + half_size_point, points.end());
    vector<float> left_part_weights(data_weights.begin(), data_weights.begin() + half_size);
    vector<float> right_part_weights(data_weights.begin() + half_size, data_weights.end());
    
    vector<float>().swap(points);
    vector<float>().swap(data_weights);

    size_int left_n_coreset = n_coreset / 2;
    size_int right_n_coreset = n_coreset - left_n_coreset;

    coreset::Points left_part_coreset(left_n_coreset, dimension);
    coreset::Points right_part_coreset(right_n_coreset, dimension);
    
    // Reduce the question to left and right part sub-questions
    left_part_coreset = compute_coreset_mr(left_part_points, left_part_weights, dimension, n_cluster, left_n_coreset, chunk_size);
    right_part_coreset = compute_coreset_mr(right_part_points, right_part_weights, dimension, n_cluster, right_n_coreset, chunk_size);
    // Merge the left and right part coreset
    coreset::Points coreset; //define coreset object
    coreset.AddPoints(left_part_coreset.GetValues(), left_part_coreset.GetWeights());
    coreset.AddPoints(right_part_coreset.GetValues(), right_part_coreset.GetWeights());
    
    return coreset;
}

