#
# 
#
sudo apt -y update
sudo apt -y upgrade

# get current directory
wrkdir=$(pwd)


# PETSC
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-configure
sudo apt -y install $(python3 firedrake-configure --show-system-packages)
sudo apt -y install python3.12-venv
export PETSC_CONFIGURE_OPTIONS="--download-hypre --download-spai  -download-exodusii --download-mumps"
git clone --depth 1 https://github.com/firedrakeproject/petsc.git
cd petsc/
python3 ../firedrake-configure --show-petsc-configure-options| xargs -L1 ./configure
make PETSC_DIR=${wrkdir}/petsc PETSC_ARCH=arch-firedrake-default all
make PETSC_DIR=${wrkdir}/petsc PETSC_ARCH=arch-firedrake-default check
cd ..

# FIREDRAKE
python3 -m venv venv-firedrake
. venv-firedrake/bin/activate
export $(python3 firedrake-configure --show-env)
pip install --no-binary h5py "firedrake @ git+https://github.com/firedrakeproject/firedrake.git#[test]"
firedrake-check 

# NIOT
git clone https://github.com/enricofacca/niot.git
cd niot/
git checkout 3dmri
pip install -e .
cd tests/
pytest test_DirichletBC.py 
