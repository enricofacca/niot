#
# bash script to run the 2024mri example 
#
# Usage: ./run.sh c dir
option=$1
c=$2
np=$3
ne=$4

label=$(date "+%Y%m%d_%H%M%S")

# copy the json file to the niot directory
cp $1 ./niot/examples/2024mri/options.json

# source the environment
source venv-firedrake/bin/activate

# move into the niot directory
cd niot

# udapte the niot package
git checkout 3dmri
git pull

# check if the niot package is installed
cd examples/2024mri/

# run the recostruction
formatted_c=$(printf "%02d" ${c})
dir="./data/subj0_box_c${formatted_c}/"
nohup mpiexec -n ${np} python ./new_reconstruct_tof.py --mri ${dir} --option options.json --n_ensemble ${ne} > submitted/${label}_c${c}.out 2> submitted/${label}_c${c}.err &
sleep 1
# %%