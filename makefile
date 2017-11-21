#
#	k-step FM-index (benchmarking for CPU and GPU)
#	Copyright (c) 2013-2017 by Alejandro Chacon	<alejandro.chacond@gmail.com>
#
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.	If not, see <http://www.gnu.org/licenses/>.
#
# PROJECT: k-step FM-index (benchmarking for CPU and GPU)
# AUTHOR(S): Alejandro Chacon <alejandro.chacond@gmail.com>
#

SHELL=bash
NVCC=nvcc
CC=gcc

CFLAGS=-O3 -m64 -I/soft/gcc-4.9.1/include
LFLAGS=-L/soft/gcc-4.9.1/lib64

PROFILE_CFLAGS=-I/soft/likwid-4.0.1/include
PROFILE_LFLAGS=-L/soft/likwid-4.0.1/lib

CUDA_CFLAGS=-O3 -m64
CUDA_LFLAGS=-L/usr/local/cuda/lib64 -lcudart 

ifndef arch
arch=big_kepler
endif

ifndef th_block
th_block=64
endif

ifndef th_sm
th_sm=64
endif

ifndef read_only
read_only=true
endif

ifndef id_device
id_device=0
endif

ifeq ($(arch),tesla)
NVCFLAGS=-gencode arch=compute_10,code=sm_10 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
endif

ifeq ($(arch),fermi)
NVCFLAGS=-gencode arch=compute_20,code=sm_20 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
endif

ifeq ($(arch),little_kepler)
NVCFLAGS=-gencode arch=compute_30,code=sm_30 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
endif

ifeq ($(arch),big_kepler)
NVCFLAGS=-gencode arch=compute_35,code=sm_35 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
ifeq ($(read_only),true)
LOADS_LDG=-DLDG
endif
endif

ifeq ($(arch),little_maxwell)
NVCFLAGS=-gencode arch=compute_50,code=sm_50 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
ifeq ($(read_only),true)
LOADS_LDG=-DLDG
endif
endif

ifeq ($(arch),big_maxwell)
NVCFLAGS=-gencode arch=compute_52,code=sm_52 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
ifeq ($(read_only),true)
LOADS_LDG=-DLDG
endif
endif

ifeq ($(arch),big_pascal)
NVCFLAGS=-gencode arch=compute_60,code=sm_60 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
ifeq ($(read_only),true)
LOADS_LDG=-DLDG
endif
endif

ifeq ($(arch),little_pascal)
NVCFLAGS=-gencode arch=compute_61,code=sm_61 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
ifeq ($(read_only),true)
LOADS_LDG=-DLDG
endif
endif

ifeq ($(arch),big_volta)
NVCFLAGS=-gencode arch=compute_70,code=sm_70 --ptxas-options=-v
CUDA_DEVICE=-DDEVICE=$(id_device)
CUDA_NUM_THREADS=-DCUDA_NUM_THREADS=$(th_block)
CUDA_NUM_THREADS_PER_SM=-DMAX_THREADS_PER_SM=$(th_sm)
ifeq ($(read_only),true)
LOADS_LDG=-DLDG
endif
endif

###########################
#	BUILDERS for FMIndexes
###########################
 
genFMI:
	$(CC) $(CFLAGS) -DK_STEPS=$(k_steps) -DNUM_CHUNK=$(d_sampling) -DNUM_COUNTERS=$(shell echo $$(( 4 ** $(k_steps) ))) -msse4.2 -o release/gfmiBaseLine_$(d_sampling)bases_$(k_steps)step common/generateIndex.c src/genFMindex.c common/common.c resources/div-tools/sssort.c resources/div-tools/trsort.c resources/divsufsort.c -lrt -fopenmp

transFmiAC:
	$(CC) $(CFLAGS) -DK_STEPS=$(k_steps) -DNUM_CHUNK=$(d_sampling) -DNUM_COUNTERS=$(shell echo $$(( 4 ** $(k_steps) ))) -msse4.2 -o release/tfmiAC_$(d_sampling)bases_$(k_steps)step common/common.c src/transformIndexAlternateCounters.c

transFmiBitmaps:
	$(CC) $(CFLAGS) -DK_STEPS=$(k_steps) -DNUM_CHUNK=$(d_sampling) -DNUM_COUNTERS=$(shell echo $$(( 4 ** $(k_steps) ))) -msse4.2 -o release/tfmiBMP_$(d_sampling)bases_$(k_steps)step common/common.c src/transformIndexBitmaps.c

fmi_builders: genFMI transFmiAC transFmiBitmaps


########################
#	CPU Searchers
########################

fmIndexSearchCPU:
	$(CC) $(CFLAGS) -DK_STEPS=$(k_steps) -DNUM_CHUNK=$(d_sampling) -DNUM_COUNTERS=$(shell echo $$(( 4 ** $(k_steps) ))) -msse4.2 -o release/fmIndexSearchCPU_$(d_sampling)bases_$(k_steps)step common/searchQueries.c src/fmIndexCPUBaseline.c common/common.c $(LFLAGS) -lrt -fopenmp

fmIndexSearchCPU-AC:
	$(CC) $(CFLAGS) -DK_STEPS=$(k_steps) -DNUM_CHUNK=$(d_sampling) -DNUM_COUNTERS=$(shell echo $$(( 4 ** $(k_steps) ))) -msse4.2 -o release/fmIndexSearchCPU_$(d_sampling)bases_$(k_steps)step-ac common/searchQueries.c src/fmIndexCPUBaseline-AltCounters.c common/common.c $(LFLAGS) -lrt -fopenmp

fmIndexSearchCPU-profile:
	$(CC) $(CFLAGS) $(PROFILE_CFLAGS) -DLIKWID_PERFMON -DK_STEPS=$(k_steps) -DNUM_CHUNK=$(d_sampling) -DNUM_COUNTERS=$(shell echo $$(( 4 ** $(k_steps) ))) -DPROFILE -msse4.2 -o release/fmIndexSearchCPU_$(d_sampling)bases_$(k_steps)step-profile common/searchQueries.c src/fmIndexCPUBaseline.c common/common.c $(LFLAGS) $(PROFILE_LFLAGS) -lrt -fopenmp -llikwid

fmIndexSearchCPU-AC-profile:
	$(CC) $(CFLAGS) $(PROFILE_CFLAGS) -DLIKWID_PERFMON -DK_STEPS=$(k_steps) -DNUM_CHUNK=$(d_sampling) -DNUM_COUNTERS=$(shell echo $$(( 4 ** $(k_steps) ))) -DPROFILE -msse4.2 -o release/fmIndexSearchCPU_$(d_sampling)bases_$(k_steps)step-ac-profile common/searchQueries.c src/fmIndexCPUBaseline-AltCounters.c common/common.c $(LFLAGS) $(PROFILE_LFLAGS) -lrt -fopenmp -llikwid


fmi_cpu_searchers: fmIndexSearchCPU fmIndexSearchCPU-AC
fmi_cpu_searchers_profile: fmIndexSearchCPU-profile fmIndexSearchCPU-AC-profile

###############################
# GPU Searchers
###############################

#	GPU TASK Searchers
fmIndexSearchGPU-task-1Step:
	$(NVCC) $(CUDA_CFLAGS) $(NVCFLAGS) -Xcompiler "-DK_STEPS=1 -DNUM_COUNTERS=4 -DNUM_CHUNK=$(d_sampling) $(LOADS_LDG) $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c src/fmIndexGPU-Task-1Step.cu -o release/fmIndexGPU-Task-1Step.o
	$(CC) $(CUDA_CFLAGS) -msse4.2 -DCUDA -DINTERLEAVING_QUERIES -DK_STEPS=1 -DNUM_COUNTERS=4 -DNUM_CHUNK=$(d_sampling) $(CUDA_LFLAGS) -o release/fmIndexGPU-1Step-task-$(d_sampling)Bases-$(arch) common/searchQueries.c src/fmIndexCPUBaseline.c release/fmIndexGPU-Task-1Step.o common/common.c -lrt -lcudart
	rm release/fmIndexGPU-Task-1Step.o

fmIndexSearchGPU-task-2Step:
	$(NVCC) $(CUDA_CFLAGS) $(NVCFLAGS) -Xcompiler "-DK_STEPS=2 -DNUM_COUNTERS=16 -DNUM_CHUNK=$(d_sampling) $(LOADS_LDG) $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c src/fmIndexGPU-Task-2Step.cu -o release/fmIndexGPU-Task-2Step.o
	$(CC) $(CUDA_CFLAGS) -msse4.2 -DCUDA -DINTERLEAVING_QUERIES -DK_STEPS=2 -DNUM_COUNTERS=16 -DNUM_CHUNK=$(d_sampling) $(CUDA_LFLAGS) -o release/fmIndexGPU-2Step-task-$(d_sampling)Bases-$(arch) common/searchQueries.c src/fmIndexCPUBaseline.c release/fmIndexGPU-Task-2Step.o common/common.c -lrt -lcudart
	rm release/fmIndexGPU-Task-2Step.o

fmIndexSearchGPU-task-2Step-AC:
	$(NVCC) $(CUDA_CFLAGS) $(NVCFLAGS) -Xcompiler "-DK_STEPS=2 -DNUM_COUNTERS=16 -DNUM_CHUNK=$(d_sampling) $(LOADS_LDG) $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c src/fmIndexGPU-Task-2Step-AltCounters.cu -o release/fmIndexGPU-Task-2Step-AltCounters.o
	$(CC) $(CUDA_CFLAGS) -msse4.2 -DCUDA -DINTERLEAVING_QUERIES -DK_STEPS=2 -DNUM_COUNTERS=16 -DNUM_CHUNK=$(d_sampling) $(CUDA_LFLAGS) -o release/fmIndexGPU-2Step-task-$(d_sampling)Bases-ac-$(arch) common/searchQueries.c src/fmIndexCPUBaseline-AltCounters.c release/fmIndexGPU-Task-2Step-AltCounters.o common/common.c -lrt -lcudart
	rm release/fmIndexGPU-Task-2Step-AltCounters.o


#	GPU COOPERATIVE Searchers
fmIndexSearchGPU-coop-1Step:
	$(NVCC) $(CUDA_CFLAGS) $(NVCFLAGS) -Xcompiler "-DK_STEPS=1 -DNUM_COUNTERS=4 -DNUM_CHUNK=$(d_sampling) -DINTERLEAVE_BMP $(LOADS_LDG) $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c src/fmIndexGPU-Coop-1Step.cu -o release/fmIndexGPU-Coop-1Step.o
	$(CC) $(CUDA_CFLAGS) -msse4.2 -DCUDA -DINTERLEAVING_QUERIES -DK_STEPS=1 -DNUM_COUNTERS=4 -DNUM_CHUNK=$(d_sampling) -DINTERLEAVE_BMP $(CUDA_LFLAGS) -o release/fmIndexGPU-1Step-$(d_sampling)Bases-$(arch) common/searchQueries.c src/fmIndexCPUBaseline.c release/fmIndexGPU-Coop-1Step.o common/common.c -lrt -lcudart
	rm release/fmIndexGPU-Coop-1Step.o

fmIndexSearchGPU-coop-2Step:
	$(NVCC) $(CUDA_CFLAGS) $(NVCFLAGS) -Xcompiler "-DK_STEPS=2 -DNUM_COUNTERS=16 -DNUM_CHUNK=$(d_sampling) -DINTERLEAVE_BMP $(LOADS_LDG) $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c src/fmIndexGPU-Coop-2Step.cu -o release/fmIndexGPU-Coop-2Step.o
	$(CC) $(CUDA_CFLAGS) -msse4.2 -DCUDA -DINTERLEAVING_QUERIES -DK_STEPS=2 -DNUM_COUNTERS=16 -DNUM_CHUNK=$(d_sampling) -DINTERLEAVE_BMP $(CUDA_LFLAGS) -o release/fmIndexGPU-2Step-$(d_sampling)Bases-$(arch) common/searchQueries.c src/fmIndexCPUBaseline.c release/fmIndexGPU-Coop-2Step.o common/common.c -lrt -lcudart
	rm release/fmIndexGPU-Coop-2Step.o

fmIndexSearchGPU-coop-2Step-AC:
	$(NVCC) $(CUDA_CFLAGS) $(NVCFLAGS) -Xcompiler "-DK_STEPS=2 -DNUM_COUNTERS=16 -DNUM_CHUNK=$(d_sampling) -DINTERLEAVE_BMP $(LOADS_LDG) $(CUDA_DEVICE) $(CUDA_NUM_THREADS) $(CUDA_NUM_THREADS_PER_SM)" -c src/fmIndexGPU-Coop-2Step-AltCounters.cu -o release/fmIndexGPU-Coop-2Step-AltCounters.o
	$(CC) $(CUDA_CFLAGS) -msse4.2 -DCUDA -DINTERLEAVING_QUERIES -DK_STEPS=2 -DNUM_COUNTERS=16 -DNUM_CHUNK=$(d_sampling) -DINTERLEAVE_BMP $(CUDA_LFLAGS) -o release/fmIndexGPU-2Step-$(d_sampling)Bases-ac-$(arch) common/searchQueries.c src/fmIndexCPUBaseline-AltCounters.c release/fmIndexGPU-Coop-2Step-AltCounters.o common/common.c -lrt -lcudart
	rm release/fmIndexGPU-Coop-2Step-AltCounters.o


#	All GPU Searchers
fmi_gpu_searchers: fmIndexSearchGPU-task-1Step fmIndexSearchGPU-task-2Step fmIndexSearchGPU-task-2Step-AC fmIndexSearchGPU-coop-1Step fmIndexSearchGPU-coop-2Step fmIndexSearchGPU-coop-2Step-AC

######################
#	INSTALL
######################

all_samplings: 
	make $(target) d_sampling=32 
	make $(target) d_sampling=64 
	make $(target) d_sampling=128
	make $(target) d_sampling=256
	make $(target) d_sampling=192 
	make $(target) d_sampling=448 
	make $(target) d_sampling=960 

all_steps: 
	make all_samplings k_steps=1
	make all_samplings k_steps=2
	make all_samplings k_steps=3
	make all_samplings k_steps=4

all_builders: 
	make all_steps target=fmi_builders

all_cpu_searchers:
	make all_steps target=fmi_cpu_searchers

all_cpu_searchers_profile:
	make all_steps target=fmi_cpu_searchers_profile

all_gpu_searchers:
	make fmi_gpu_searchers d_sampling=64
	make fmi_gpu_searchers d_sampling=192 
	make fmi_gpu_searchers d_sampling=448 
	make fmi_gpu_searchers d_sampling=960 

all:	all_builders      \
      all_cpu_searchers \
	    all_gpu_searchers \
	    install

clean:
	rm -f release/*
	rm -f bin/*

install:
	mkdir -p bin
	find release -perm /a+x -type f -exec mv {} bin \;
	cp resources/genreads.py bin

