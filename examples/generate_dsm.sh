#!/bin/bash

# Function to generate a random sparse DSM matrix of given size and sparsity
generate_sparse_dsm() {
    local size=$1
    local sparsity=$2
    local filename=$3

    echo "Generating sparse DSM of size ${size}x${size} with sparsity ${sparsity} in file ${filename}"

    # Create the header
    header=","
    # Add the rows with random values
    for i in $(seq 1 $size); do
        header+="Element${i},"
    done
    # Create the CSV file and add the header
    echo ${header%,} > $filename

    # Add the rows with random values
    for i in $(seq 1 $size); do
        row="Element${i},"
        for j in $(seq 1 $size); do
            if [ $i -eq $j ]; then
                row+="1,"
            else
                if (( $(echo "$RANDOM / 32767.0 < $sparsity" | bc -l) )); then
                    row+="1,"
                else
                    row+="0,"
                fi
            fi
        done
        echo ${row%,} >> $filename
    done
}

# Generate a sparse DSM 
size=1000
sparsity=0.3
filename="sparse_dsm_${size}x${size}_sparsity_${sparsity}.csv"
generate_sparse_dsm $size $sparsity $filename

echo "Sparse DSM generated in file ${filename}"