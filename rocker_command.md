rocker --privileged --user --nvidia --volume /home/nlitz88/repos/f1tenth_lab8/:/lab8/ -- nvcr.io/nvidia/pytorch:23.04-py3

Note that when installing pip dependenices, make sure to install as root, not as
your current user. For instance, when installing pycuda for the detection
script, you want it to be installed in the "global" python workspace. Of course,
using a virtual environment is the way to go here, but because we're using a
development container, we're kinda of treating this container's global python
installation as our virtual environment.