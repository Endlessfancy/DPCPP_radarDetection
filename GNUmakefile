# Makefile for GNU Make

default: run_all

all: run_all

run_all: radardetection 
	./radardetection

DPCPP_OPTS = -DMKL_ILP64 -I${MKLROOT}/include -qmkl=parallel

radardetection: radardetecion.cpp
	icpx $< -fsycl -o $@ $(DPCPP_OPTS)

clean:
	-rm -f radardetection

.PHONY: run_all clean all
