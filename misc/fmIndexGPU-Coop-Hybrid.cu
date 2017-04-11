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

#include "../common/common-search.h"
#include "../common/common-cuda.h"

#define SIZE_ENTRIES_BUFFER      2

typedef struct {
  uint data[SIZE_ENTRY];
} bitcnt_t;

typedef struct {
  uint4 loadData[SIZE_WARP];
} buffer_t;

extern "C"
size_t sizeOfEntry(){
  return(sizeof(bitcnt_t));
}

inline __device__ uint countBitmap(uint bitmap, int shift)
{
  uint mask;
  mask = 0xFFFFFFFF << (32-shift);
  mask = (shift>32) ? 0xFFFFFFFF : mask;
  mask = (shift>0) ? mask : 0x0;
  return (__popc(bitmap & mask));
}

inline __device__ uint computeBitmaps(uint4 bitmap, uint interval, uint indexBase, uint localRequestThreadIdx)
{
  uint bitmapA, bitmapB;
  uint bit0 = indexBase & 0x01;
  uint bit1 = indexBase & 0x02;
  int shift = (interval % NUM_CHUNK) - (localRequestThreadIdx * 64);
  
  bitmap.x = bit0 ? bitmap.x : ~bitmap.x;  
  bitmap.y = bit1 ? bitmap.y : ~bitmap.y;

  bitmap.z = bit0 ? bitmap.z : ~bitmap.z;  
  bitmap.w = bit1 ? bitmap.w : ~bitmap.w;

  bitmapA = bitmap.x & bitmap.y;
  bitmapB = bitmap.z & bitmap.w;

  bitmapA = countBitmap(bitmapA, shift);
  bitmapB = countBitmap(bitmapB, shift - 32);
  
  return (bitmapA + bitmapB);
}

inline __device__ uint selectCounter(uint4 counters, uint indexBase)
{
  uint counter;
  if (indexBase == 0) counter = counters.x;
  if (indexBase == 1) counter = counters.y;
  if (indexBase == 2) counter = counters.z;
  if (indexBase == 3) counter = counters.w;
  return(counter);
}

inline __device__ uint intraWarpAddReduce(uint values, uint localThreadIdx)
{
  for (int i=1; i<THREADS_PER_REQUEST; i*=2){
    int n = __shfl_down((int) values, i, 32);
      values += n;
  }
  values = __shfl((int) values, (int) ((localThreadIdx % REQUESTS_PER_WARP) * THREADS_PER_REQUEST));
  return(values);
}

inline __device__ uint localCountReduce(uint4 loadData, uint sharedInterval, uint sharedIndexBase, uint localRequestThreadIdx)
{
  uint reducedEntry;  
  if(localRequestThreadIdx != (THREADS_PER_REQUEST - 1))
    reducedEntry = computeBitmaps(loadData, sharedInterval, sharedIndexBase, localRequestThreadIdx);
  else 
    reducedEntry = selectCounter(loadData, sharedIndexBase);
  return (reducedEntry);
}

inline __device__ void loadEntriesInBuffer(buffer_t *loadDataBuffer, bitcnt_t *indexFM, uint *sharedInterval, uint localWarpIdx, uint localRequestThreadIdx, uint localThreadIdx)
{
  uint4 loadData0, loadData1;
  loadData0 = ((uint4 *) (indexFM[sharedInterval[0]/NUM_CHUNK].data))[localRequestThreadIdx];
  loadData1 = ((uint4 *) (indexFM[sharedInterval[1]/NUM_CHUNK].data))[localRequestThreadIdx];
  if( clock() == 0 ) loadData1.x+=1;
  loadDataBuffer[localWarpIdx * SIZE_ENTRIES_BUFFER + 0].loadData[localThreadIdx] = loadData0;
  loadDataBuffer[localWarpIdx * SIZE_ENTRIES_BUFFER + 1].loadData[localThreadIdx] = loadData1;
}

inline __device__ uint compactIntervals(uint next_interval, uint reducedEntry, int i, int posBuffer, uint localThreadIdx)
{
  if(((REQUESTS_PER_WARP * (i + posBuffer)) <= localThreadIdx) && (localThreadIdx < (REQUESTS_PER_WARP * (i + 1 + posBuffer))))
    next_interval = reducedEntry;
  return (next_interval);
}

inline __device__ uint processQuery(uint word, uint intraStep)
{
  word <<= (8 * intraStep);
  uint bit1 = word & 0x02000000;
  uint flg2 = word & 0x01000000;
  uint flg3 = flg2 ^ 0x01000000;
  uint bit0 = bit1 ? flg3 : flg2;
  uint indexBase = (bit1 | bit0) >> 24;
  return (indexBase);
}

__global__ void searchIndexKernel(uint bwtsize, uint chunk, bitcnt_t *indexFM, 
                  uint dollarPositionBWT, uint dollarBaseBWT, uint modposdollarBWT, 
                  uint numQueries, uint sizeQuery, char *queries, uint *results)
{ 
  __shared__ buffer_t loadDataBuffer[NUM_WARPS_PER_BLOCK * SIZE_ENTRIES_BUFFER];

  uint globalThreadIdx = blockIdx.x * MAX_THREADS_PER_SM + threadIdx.x;

  if ((threadIdx.x < MAX_THREADS_PER_SM) && (globalThreadIdx < (numQueries * 2))){

    unsigned int *groupQueries = NULL;
    
    uint globalWarpIdx          = globalThreadIdx / SIZE_WARP;
    uint localWarpIdx           = threadIdx.x / SIZE_WARP;
    uint localThreadIdx         = threadIdx.x % SIZE_WARP;
    uint localRequestThreadIdx  = localThreadIdx % THREADS_PER_REQUEST; 
        
    uint interval = (localThreadIdx % 2) ? bwtsize : 0;
    uint next_interval;
       
    groupQueries = (unsigned int *) (queries + ((globalWarpIdx / 2) * SIZE_WARP * sizeQuery));
    groupQueries = (globalWarpIdx % 2) ? groupQueries + 16 : groupQueries;
    
    #pragma unroll 1
    for(uint queryWord = 0; queryWord < sizeQuery/4; queryWord++) {
      uint word = groupQueries[queryWord * SIZE_WARP + (localThreadIdx / 2)] >> 1;
      #pragma unroll 1
      for(int j = 0; j < 4; j++) {
        uint indexBase = processQuery(word, j);
        #pragma unroll 1
        for(int i = 0; i < 8; i += SIZE_ENTRIES_BUFFER){
          uint nSharedEntry, sharedInterval[SIZE_ENTRIES_BUFFER], sharedIndexBase[SIZE_ENTRIES_BUFFER];
          for(int posBuffer = 0; posBuffer < SIZE_ENTRIES_BUFFER; posBuffer++){
            nSharedEntry                = ((REQUESTS_PER_WARP * (i + posBuffer)) + (localThreadIdx / THREADS_PER_REQUEST));
            sharedInterval[posBuffer]   = (uint) __shfl((int) interval, (int) nSharedEntry);
            sharedIndexBase[posBuffer]  = (uint) __shfl((int) indexBase, (int) nSharedEntry);
          }
          loadEntriesInBuffer(loadDataBuffer, indexFM, sharedInterval, localWarpIdx, localRequestThreadIdx, localThreadIdx);          
          for(int posBuffer = 0; posBuffer < SIZE_ENTRIES_BUFFER; posBuffer++){
            uint4 FM_Entry = loadDataBuffer[localWarpIdx * SIZE_ENTRIES_BUFFER + posBuffer].loadData[localThreadIdx];
            uint reducedEntry;
            reducedEntry = localCountReduce(FM_Entry, sharedInterval[posBuffer], sharedIndexBase[posBuffer], localRequestThreadIdx);
            reducedEntry = intraWarpAddReduce(reducedEntry, localThreadIdx);
            next_interval = compactIntervals(next_interval, reducedEntry, i, posBuffer, localThreadIdx);
          }
       }
                
       if(modposdollarBWT == (interval / NUM_CHUNK))
         next_interval = ((indexBase == dollarBaseBWT) && (interval > dollarPositionBWT)) ? next_interval - 1 : next_interval;  
       interval = next_interval;
      }
    }
    results[globalThreadIdx] = interval;
  }
}

extern "C"
void launchKernelAsync(fmi_t *fmi, qrys_t *qrys, int numQueries, unsigned char *queries, uint *results, cudaStream_t stream){

  uint blocks = ((numQueries * 2) / MAX_THREADS_PER_SM) + (((numQueries * 2) % MAX_THREADS_PER_SM) ? 1 : 0);
  uint threads = CUDA_NUM_THREADS;

  searchIndexKernel<<< blocks, threads, 0, stream >>>(fmi->bwtsize, fmi->chunk, (bitcnt_t *) fmi->d_index, 
                              fmi->h_dollarPositionBWT[0], fmi->h_dollarBaseBWT[0], 
                              fmi->h_modposdollarBWT[0], numQueries, qrys->size,
                              queries, results);
}


