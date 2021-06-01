/*
Merge and Reduce version of Coreset method. 
Utilize the key property: 'Composable' to paralize the computation of coreset, 
and deal with the situation that the input data is ouersize the storage of GPU memory.
*/

#include "coreset.cu"
//typedef unsigned long long int size_int;


coreset::FlatPoints
compute_coreset_mr(vector<float> &points, vector<float> &data_weights,
    unsigned int dimension, unsigned int n_cluster, size_int n_coreset, unsigned int chunk_size, size_int li, size_int ri){
	
    size_int n = ri - li + 1;

    if(n <= chunk_size){
        coreset::FlatPoints data_chunk(n, dimension);
        vector<float> chunk_points(points.begin() + li*dimension, points.begin() + (ri + 1)*dimension);
        vector<float> chunk_weights(data_weights.begin() + li, data_weights.begin() + ri + 1);

        data_chunk.FillPoints(chunk_points, chunk_weights);
        return data_chunk;  // weighted-points
    }

    size_int half_size = n / 2;

    size_int left_li = li;
    size_int left_ri = li + half_size;
    size_int right_li = li + half_size + 1;
    size_int right_ri = ri;

    coreset::FlatPoints left_part;
    coreset::FlatPoints right_part;
    // Reduce the question to left and right part sub-questions
    left_part = compute_coreset_mr(points, data_weights, dimension, n_cluster, n_coreset, chunk_size, left_li, left_ri);
    right_part = compute_coreset_mr(points, data_weights, dimension, n_cluster, n_coreset, chunk_size, right_li, right_ri);
    // Merge the left and right part coreset
    coreset::FlatPoints coreset; //define coreset object
    coreset.AddPoints(dimension, left_part.GetValues(), left_part.GetWeights());
    coreset.AddPoints(dimension, right_part.GetValues(), right_part.GetWeights());
    vector <float> values = coreset.GetValues();
    vector <float> weights = coreset.GetWeights();
    // Compute and return the coreset of merged previous layer coreset
    return compute_coreset_flat(values, weights, dimension, n_cluster, n_coreset);

}


/*
coreset::FlatPoints
compute_coreset_mr(vector<float> &points, vector<float> &data_weights, 
    unsigned int dimension, unsigned int n_cluster, size_int n_coreset, unsigned int chunk_size, size_int li, size_int ri){*/
    /*
	points: original data points
	data_weights: weights of points
	chunk_size: the size of the chunk that we conduct the computation of coreset	
	li, ri: the left and right index that we applied to get the current points in the original points
    */
/*
    size_int n = ri - li + 1;  
    
    if(n <= chunk_size){
    	return compute_coreset_flat(values, weights, dimension, n_cluster, n_coreset);
    }
    else{
	return 	recurse(points, data_weights, dimension, n_cluster, n_coreset, chunk_size, li, ri);
    }
}
*/
