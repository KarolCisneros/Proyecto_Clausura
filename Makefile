# Compilación del proyecto Bolsa de Palabras (serial + MPI)
#   make          -> construye build/bow_serial y build/bow_mpi
#   make clean    -> borra build/ y output/
MPICXX   ?= mpicxx
SERIALCXX?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
LDLIBS    = -lcurl

all: build/bow_serial build/bow_mpi

build:
	mkdir -p build

build/bow_serial: source/bow_serial.cpp source/bow_core.hpp | build
	$(SERIALCXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

build/bow_mpi: source/bow_mpi.cpp source/bow_core.hpp | build
	$(MPICXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -rf build output

.PHONY: all clean
