# readme 
The CPU-GPU version is in usm branch.

This main branch is the previous version, you can ignore this version.
this file is based on the vector add file. Now the cmakefile and linkfile haven't changed yet. So the code for radar detection part is in the vector-add-buffers.cpp file. You don't need to change the cmake file or any other file. Just directly programming in the vector-add-buffers.cpp.

To compile:
1. make build directory
2. cd the build file
3. command:
```
cmake ..
```
4. command:
```
make cpu-gpu
```
5. move the fhy_s.bin and fhy_direct.bin file into the build file.
6. run ./vector-add-buffers

! now you can only run this in the local machine. Devcloud still has some bugs that I can't handle them.
