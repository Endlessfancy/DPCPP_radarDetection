# Makefile for NMAKE

default: run_all

all: run_all

run_all: radardetection 
	./radardetection

DPCPP_OPTS=-DMKL_ILP64 -I"%MKLROOT%\include" /Qmkl OpenCL.lib /EHsc

radardetection: radardetection.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

clean:
	del /q radardetection.exe 
	del /q radardetection.exp 

pseudo: run_all clean all

