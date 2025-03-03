#
# run with echo Y Y Y | ./install_firedrake_and_niot.sh
#
sudo apt update
sudo apt upgrade
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/firedrake-configure
sudo apt install $(python3 firedrake-configure --show-system-packages)
export PETSC_CONFIGURE_OPTIONS="--download-hypre --download-spai"
git clone --depth 1 https://github.com/firedrakeproject/petsc.git
cd petsc/
python3 ../firedrake-configure --show-petsc-configure-options
python3 ../firedrake-configure --show-petsc-configure-options| xargs -L1 ./configure
make PETSC_DIR=/home/ubuntu/petsc PETSC_ARCH=arch-firedrake-default all
make PETSC_DIR=/home/ubuntu/petsc PETSC_ARCH=arch-firedrake-default check
make PETSC_DIR=/home/ubuntu/petsc PETSC_ARCH=arch-firedrake-default check
cd ..
sudo apt install python3.12-venv
python3 -m venv venv-firedrake
. venv-firedrake/bin/activate
export $(python3 firedrake-configure --show-env)
pip install --no-binary h5py "firedrake @ git+https://github.com/firedrakeproject/firedrake.git#[test]"
firedrake-check 
git clone https://github.com/enricofacca/niot.git
cd niot/
git checkout 3dmri
pip install -e .
cd tests/
pytest test_DirichletBC.py 
