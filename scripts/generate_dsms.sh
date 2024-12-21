#!/bin/bash

# Function to generate a random DSM matrix of given size
generate_dsm() {
    local size=$1
    local filename=$2

    echo "Generating DSM of size ${size}x${size} in file ${filename}"

    # Create the CSV file and add the header
    echo ","$(seq -s, 1 $size) > $filename

    # Add the rows with random values
    for i in $(seq 1 $size); do
        row="${i},"
        for j in $(seq 1 $size); do
            if [ $i -eq $j ]; then
                row+="0,"
            else
                row+=$((RANDOM % 2))","
            fi
        done
        echo ${row%,} >> $filename
    done
}

# Directory to store the generated DSMs
output_dir="dsms"
mkdir -p $output_dir

# Generate DSMs of sizes from 3x3 to 10x10
filename="${output_dir}/dsm_${size}x${size}.csv"
generate_dsm $1 $2