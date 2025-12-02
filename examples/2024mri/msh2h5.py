from firedrake import Mesh, CheckpointFile, COMM_WORLD
from firedrake.petsc import PETSc
import time
import os
import argparse

def convert_msh_to_h5(input_directory, out_directory):
    start = time.time() 
    PETSc.Sys.Print(f"Mesh loading started...", end="")
    mesh = Mesh(os.path.join(input_directory,"brain_main.msh"))
    PETSc.Sys.Print(f" completed in {time.time()-start:.2e} s")
    print("New boundary markers (should include 99):", mesh.topology.exterior_facets.unique_markers)
    nproc = PETSc.COMM_WORLD.getSize()

    start = time.time() 
    h5_filename = os.path.join(out_directory, f"brain_main_{nproc:04d}.h5")
    PETSc.Sys.Print(f"Saving mesh to {h5_filename}...", end="")
    with CheckpointFile(h5_filename, 'w', comm=COMM_WORLD) as afile:
            afile.save_mesh(mesh,"relabeled_mesh")
            PETSc.Sys.Print(f" completed in {time.time()-start:.2e} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .msh mesh to .h5 format.")
    parser.add_argument("-i","--input_directory", type=str, help="Directory containing the input .msh file.")
    parser.add_argument("-o","--out_directory", type=str, help="Directory to save the output .h5 file.")
    args = parser.parse_args()

    input_directory = args.input_directory
    out_directory = args.out_directory

    convert_msh_to_h5(input_directory, out_directory)