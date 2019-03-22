#!/bin/bash
fileName='out'
echo "Time,CORE,RAM,CPU,MEMORY" >> results/$fileName.csv
for i in {1..10}
do
    ./performancer.sh $fileName
done