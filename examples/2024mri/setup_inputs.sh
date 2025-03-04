t=$1
python connected_components_tof.py --mri data/subj0_c01/ --threshold ${t};

for f in data/subj0_c01/*nii.gz;
do
    for c in 2 4 8;
    do
	bn=$(basename $f);
	python resize_nifti.py --c ${c} --input ${f} --output data/subj0_c0${c}/${bn};
    done;
done
