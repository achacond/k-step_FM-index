/*
 *  k-step FM-index (benchmarking for CPU and GPU)
 *  Copyright (c) 2011-2017 by Alejandro Chacon  <alejandro.chacond@gmail.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * PROJECT: k-step FM-index (benchmarking for CPU and GPU)
 * AUTHOR(S): Alejandro Chacon <alejandro.chacond@gmail.com>
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/common.h"
#include <cuda_runtime.h>
#include <cuda.h>

#if defined(K_STEPS) || defined(NUM_CHUNK) || defined(NUM_COUNTERS)
#else
  #define K_STEPS         1
  #define NUM_CHUNK       64
  #define NUM_COUNTERS    4
#endif

#if defined(CUDA_NUM_THREADS) || defined(MAX_THREADS_PER_SM) || defined(DEVICE)
#else
  #define CUDA_NUM_THREADS    128
  #define MAX_THREADS_PER_SM  128
  #define DEVICE              0
#endif

#define NUM_BITMAPS         (NUM_CHUNK / 32)
#define BITS_PER_BASE       2

#define SIZE_WARP           32
#define NUM_WARPS_PER_BLOCK (1 + ((CUDA_NUM_THREADS - 1) / SIZE_WARP))

#define SIZE_DATA           ((BITS_PER_BASE * NUM_BITMAPS * K_STEPS) + NUM_COUNTERS)
#define BYTES_PER_DATA      (((BITS_PER_BASE * NUM_BITMAPS * K_STEPS) + NUM_COUNTERS) * 4)
#define ELEM_PER_WARP       (SIZE_WARP / SIZE_DATA)
#define SIZE_VECTOR_TYPE    4

typedef struct {
  uint4 entry[((BITS_PER_BASE * NUM_BITMAPS * K_STEPS) + NUM_COUNTERS) / SIZE_VECTOR_TYPE];
} bitcnt_t;

typedef struct {
  uint32_t steps;
  uint32_t bwtsize;
  uint32_t ncounters;
  uint32_t nentries;
  uint32_t chunk;
  uint32_t nbitmaps;
  uint32_t *h_dollarPositionBWT;
  uint32_t *h_dollarBaseBWT;
  uint32_t *h_modposdollarBWT;
  bitcnt_t *h_index;
  uint32_t *d_dollarPositionBWT;
  uint32_t *d_dollarBaseBWT;
  uint32_t *d_modposdollarBWT;
  bitcnt_t *d_index;
} fmi_t;


extern "C" 
static void HandleError( cudaError_t err, const char *file,  int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString(err),  file, line );
      exit( EXIT_FAILURE );
  }
}

inline __device__ uint32_t selectCounter(uint4 counters, uint32_t indexCounter)
{
  uint32_t counter;
  if (indexCounter == 0) counter = counters.x;
  if (indexCounter == 1) counter = counters.y;
  if (indexCounter == 2) counter = counters.z;
  if (indexCounter == 3) counter = counters.w;
  return(counter);
}

inline __device__ uint32_t selectBitmap0(uint4 bitmaps, uint32_t indexBitmap)
{
  uint32_t bitmap;
  if (indexBitmap == 0) bitmap = bitmaps.x;
  if (indexBitmap == 1) bitmap = bitmaps.z;
  return(bitmap);
}

inline __device__ uint32_t selectBitmap1(uint4 bitmaps, uint32_t indexBitmap)
{
  uint32_t bitmap;
  if (indexBitmap == 0) bitmap = bitmaps.y;
  if (indexBitmap == 1) bitmap = bitmaps.w;
  return(bitmap);
}

__global__ void searchIndexKernel(uint32_t bwtsize, uint32_t chunk, bitcnt_t *indexFM, 
                                  uint32_t dollarPositionBWT, uint32_t dollarBaseBWT, uint32_t modposdollarBWT, 
                                  uint32_t numQueries, uint32_t sizeQuery, char *queries, uint32_t *results)
{ 
  uint32_t *groupQueries = NULL;
  uint32_t j, n;
  uint32_t globalThreadIdx, localThreadIdx, globalWarpIdx;
  
  uint32_t indexBitmap, indexBase, flg2, flg3;
  uint32_t word, queryWord, bitCount, indexCounter, bitmap = 0xFFFFFFFF, bitmapShifted, mask = 0xFFFFFFFF;
  uint32_t bitmap1, bitmap0, not_bitmap1, not_bitmap0;
  uint32_t bit0, bit1;
  uint32_t nQueryWords = sizeQuery/4;
  int32_t  shift;
  uint4    buffer;

  globalThreadIdx = blockIdx.x * MAX_THREADS_PER_SM + threadIdx.x;

  if ((threadIdx.x < MAX_THREADS_PER_SM) && (globalThreadIdx < (numQueries * 2))){
    globalWarpIdx = globalThreadIdx / SIZE_WARP;
    localThreadIdx = threadIdx.x % SIZE_WARP;
        
    uint32_t interval = (localThreadIdx % 2) ? bwtsize : 0;

    groupQueries = (uint32_t *) (queries + ((globalWarpIdx / 2) * SIZE_WARP * sizeQuery));
    groupQueries = (globalWarpIdx % 2) ? groupQueries + 16 : groupQueries;
    
    for(queryWord = 0; queryWord < nQueryWords; queryWord++) {
      word = groupQueries[queryWord * SIZE_WARP + (localThreadIdx / 2)] >> 1;
      
      for(j = 0; j < 4/K_STEPS; j++) {
        bit1 = word & 0x02000000;
        flg2 = word & 0x01000000;
        flg3 = flg2 ^ 0x01000000;
        bit0 = bit1 ? flg3 : flg2;
        indexBase = (bit1 | bit0) >> 24;
        word <<= 8;

        indexCounter = interval / NUM_CHUNK;
        shift = interval % NUM_CHUNK;
        bitCount = 0;
        
        for(n = 0; n < NUM_BITMAPS/2; n++){
          // Memory accesses of 16 Bytes
          buffer = indexFM[indexCounter].entry[n];
          for(indexBitmap = 0; indexBitmap < 2; indexBitmap++){
            bitmapShifted = mask << (32 - shift);
            bitmap = (shift > 32) ? mask : bitmapShifted;
            bitmap = (shift > 0) ? bitmap : 0x0;

            bitmap0 = selectBitmap0(buffer, indexBitmap);
            bitmap1 = selectBitmap1(buffer, indexBitmap);
            not_bitmap0 = ~bitmap0;
            not_bitmap1 = ~bitmap1;
            bitmap0 = bit0 ? bitmap0 : not_bitmap0;
            bitmap1 = bit1 ? bitmap1 : not_bitmap1;
            bitmap &= (bitmap0 & bitmap1);   

            bitCount += __popc(bitmap);
            shift -= 32;
          }
        }

        if(modposdollarBWT==indexCounter)
          bitCount = ((indexBase == dollarBaseBWT) && (interval > dollarPositionBWT)) ? bitCount - 1 : bitCount;

        buffer   = indexFM[indexCounter].entry[n];
        interval = selectCounter(buffer, indexBase) + bitCount;
      }
    }
    results[globalThreadIdx] = interval;
  }
}

extern "C" 
void searchIndexGPU(void *index, void *dataqueries, void *resIntervals) 
{
  fmi_t *fmi = (fmi_t *) index;
  res_t *res=(res_t *) resIntervals;
  qrys_t *qrys=(qrys_t *) dataqueries;
  int32_t blocks=((qrys->num * 2) / MAX_THREADS_PER_SM) + (((qrys->num * 2) % MAX_THREADS_PER_SM) ? 1 : 0);
  int32_t threads=CUDA_NUM_THREADS;

  printf("Bloques: %d - Th_block %d - Th_sm %d\n", blocks, threads, MAX_THREADS_PER_SM);

  searchIndexKernel<<<blocks,threads>>>(fmi->bwtsize, fmi->chunk, fmi->d_index, 
                      fmi->h_dollarPositionBWT[0], fmi->h_dollarBaseBWT[0], 
                      fmi->h_modposdollarBWT[0], qrys->num, qrys->size,
                      qrys->d_queries, res->d_results);
  cudaThreadSynchronize();
}

extern "C" 
int32_t transferCPUtoGPU(void *index, void *dataqueries, void *resIntervals)
{
  fmi_t *fmi = (fmi_t *) index;
  res_t *res = (res_t *) resIntervals;
  qrys_t *qrys = (qrys_t *) dataqueries;
 
  CUDA_HANDLE_ERROR(cudaSetDevice(DEVICE));

  // allocate & transfer FMIndex to GPU
  CUDA_HANDLE_ERROR(cudaMalloc((void**)&fmi->d_index, (fmi->nentries)*sizeof(bitcnt_t)));
  CUDA_HANDLE_ERROR(cudaMemcpy(fmi->d_index, fmi->h_index, (fmi->nentries)*sizeof(bitcnt_t), cudaMemcpyHostToDevice));

  // allocate & transfer Queries to GPU
  CUDA_HANDLE_ERROR(cudaMalloc((void**)&qrys->d_queries, (qrys->num)*(qrys->size)*sizeof(unsigned char)));
  CUDA_HANDLE_ERROR(cudaMemcpy(qrys->d_queries, qrys->h_queries, (qrys->num)*(qrys->size)*sizeof(unsigned char), cudaMemcpyHostToDevice));

  // allocate Results
  CUDA_HANDLE_ERROR(cudaMalloc((void**)&res->d_results, 2*(res->num)*sizeof(uint32_t)));
  CUDA_HANDLE_ERROR(cudaMemset(res->d_results, 0, 2*(res->num)*sizeof(uint32_t)));

  return (SUCCESS);
}

extern "C" 
int32_t transferGPUtoCPU(void *resIntervals)
{
  res_t *res = (res_t *) resIntervals;
  CUDA_HANDLE_ERROR(cudaMemcpy(res->h_results, res->d_results, 2 * res->num * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  return (SUCCESS);
}

extern "C" 
int32_t freeIndexGPU(void **index) 
{  
  fmi_t *fmi = (fmi_t *) (*index);  
  if(fmi->d_dollarPositionBWT != NULL){
    cudaFree(fmi->d_dollarPositionBWT);
    fmi->d_dollarPositionBWT=NULL;
  }
  if(fmi->d_dollarBaseBWT != NULL){
    cudaFree(fmi->d_dollarBaseBWT);
    fmi->d_dollarBaseBWT=NULL;
  }
  if(fmi->d_modposdollarBWT != NULL){
    cudaFree(fmi->d_modposdollarBWT);
    fmi->d_modposdollarBWT=NULL;
  }
  if(fmi->d_index != NULL){
    cudaFree(fmi->d_index);
    fmi->d_index=NULL;
  }
  return(SUCCESS);
}

extern "C" 
int32_t freeQueriesGPU(void **dataqueries) 
{  
  qrys_t *qrys = (qrys_t *) (*dataqueries);  
  if(qrys->d_queries != NULL){
    cudaFree(qrys->d_queries);
    qrys->d_queries=NULL;
  }
  return(SUCCESS);
}

extern "C" 
int32_t freeResultsGPU(void **resIntervals) 
{  
  res_t *res = (res_t *) (*resIntervals);  
  if(res->d_results != NULL){
    cudaFree(res->d_results);
    res->d_results=NULL;
  }
  return(SUCCESS);
}

