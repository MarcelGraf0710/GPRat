#!/bin/bash
#
#SBATCH --job-name=gprat
#SBATCH --output=output_gprat.txt
#SBATCH --time=48:00:00

source ~/Setup_Scripts/setup_breyerml.sh

if [[ "$HOSTNAME" == "simcl1n1" || "$HOSTNAME" == "simcl1n2" ]]; then

    ./execute-benchmark.sh no no no yes gpu nvidia cuda
    ./execute-benchmark.sh no no no yes gpu nvidia sycl

elif [[ "$HOSTNAME" == "simcl1n3" ]]; 
then

    ./execute-benchmark.sh no no no yes gpu amd

elif [[ "$HOSTNAME" == "simcl1n4" ]]; 
then

    ./execute-benchmark.sh no no no yes cpu cpu

fi