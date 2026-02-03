from firedrake import Mesh, CheckpointFile, COMM_WORLD
from firedrake.petsc import PETSc
import time
import os
import argparse

def convert_msh_to_h5(input_msh, output_h5):
    start = time.time() 
    PETSc.Sys.Print(f"Mesh loading started...", end="")
    mesh = Mesh(input_msh)
    mesh.name = "mesh"
    PETSc.Sys.Print(f" completed in {time.time()-start:.2e} s")
    print("New boundary markers (should include 99):", mesh.topology.exterior_facets.unique_markers)
    nproc = PETSc.COMM_WORLD.getSize()

    start = time.time() 
    h5_filename = output_h5
    PETSc.Sys.Print(f"Saving mesh to {output_h5}...", end="")
    with CheckpointFile(h5_filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_mesh(mesh,"mesh")
        PETSc.Sys.Print(f" completed in {time.time()-start:.2e} s")
    
    num_cells = 0
    num_vertices = 0
    num_facets = 0
    for i in range(nproc):
        if i == PETSc.COMM_WORLD.getRank():
            PETSc.Sys.Print(f"Process {i} saved mesh to {h5_filename}")
            num_cells += mesh.num_cells()
            num_vertices += mesh.num_vertices()
            num_facets += mesh.num_facets()


    PETSc.Sys.Print(f'Cells: {num_cells}'
                    + f' Nodes: {num_vertices}'
                    + f' Facets: {num_facets}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .msh mesh to .h5 format.")
    parser.add_argument("-i","--input", type=str, help="Directory containing the input .msh file.")
    parser.add_argument("-o","--out", type=str, help="Directory to save the output .h5 file.")
    args = parser.parse_args()

    input_file = args.input
    out_file = args.out

    convert_msh_to_h5(input_file, out_file)