# niot
Network Inpainting via Optimal Transport

# Installation
The code is written in python and it requires the package [Firedrake](https://www.firedrakeproject.org) to be installed. See [this page](https://www.firedrakeproject.org/download.html) for installation guidance.

The authors suggests the installation via Docker, which is a containerazion platform that simplies the installation of software. First [install docker](https://docs.docker.com/engine/install/) and then "download" the firedrake package
```
docker pull firedrakeproject/firedrake
```

Last, clone this repository with
```
git clone https://github.com/enricofacca/niot.git
``` 

# Usage
First move in the directory were the reposity has been clone and run the Firedrake image. In Linux run the command
```
docker run -ti -v $(pwd):/home/firedrake/shared firedrakeproject/firedrake
```
It will run the Docker container and create a directory name "shared", that is a "mirror" of the directory containg the cloned repository (the current work directoy, this is why there is the $(pwd) argment).

Any file change done inside the running container can be seen from outside, and viceversa. Hence, we suggests to open a second terminal and use you favorite editor, visulatization tools (VisIt, Paraview, matplotlib), etc. to work directly on the files within the cloned repository, while you can run all NIOT programs within the Docker container. You find an example in the Usage section below.

But first, you  need activate the Firedrake enviroment in the Docker image.
In Linux, run the command
```
source firedrake/bin/activate
```

# Tests / Examples
Move in the test directory ad run
```
python test.py
```
It will run a program taking a synthetic net, first corrupting it and then recosntructing it. Other example are present and described in the same directory.

# Authors
Enrico Facca, enrico.facca@uib.no : Departement of Mathematics, University of Bergen, Bergen, Norway.

