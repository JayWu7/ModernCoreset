#!/bin/bash

#repeat the experiments with different coreset_size seeting: (1,2,5,10)*(100,1000,10000)
filename=USCensus1990.data
dimension=68
for coreset_size in 100 200 500 800 1000 2000 5000 8000 10000 20000 50000 80000 100000
do
echo "*************************************************" 
bash ./run.sh $filename $coreset_size $dimension

done

