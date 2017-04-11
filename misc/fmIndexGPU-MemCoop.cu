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
  

typedef struct {
  uint data[SIZE_ENTRY];
} bitcnt_t;

typedef union {
  uint  entry[SIZE_WARP][SIZE_ENTRY];
  uint4 entryV4[SIZE_WARP][SIZE_ENTRY / SIZE_VECTOR_TYPE];
} s_entry_t;

typedef struct {
  uint interval[SIZE_WARP];
} s_interval_t;



extern "C"
size_t sizeOfEntry(){
  return(sizeof(bitcnt_t));
}

inline __device__ uint countBitmaps(uint interval, uint localWarpIdx, uint localThreadIdx, uint bit0, uint bit1, s_entry_t *s_indexFM)
{
  int shift = interval % NUM_CHUNK;
  uint bitCount = 0;
  uint mask, bitmap1, bitmap0, n;
  uint not_bitmap0, not_bitmap1;

  for(n=0; n<NUM_BITMAPS; n++){
    bitmap0 = s_indexFM[localWarpIdx].entry[localThreadIdx][n];
    bitmap1 = s_indexFM[localWarpIdx].entry[localThreadIdx][NUM_BITMAPS + n];
    mask = 0xFFFFFFFF << (32-shift);
    mask = (shift>32) ? 0xFFFFFFFF : mask;
    mask = (shift>0) ? mask : 0x0;

    not_bitmap0 = ~bitmap0;
    not_bitmap1 = ~bitmap1;
    bitmap0 = bit0 ? bitmap0 : not_bitmap0;
    bitmap1 = bit1 ? bitmap1 : not_bitmap1;
        
    bitCount += __popc((bitmap0 & bitmap1) & mask);
    shift -= 32;
  }
  return (bitCount);
}


__global__ void searchIndexKernel(uint bwtsize, uint chunk, bitcnt_t *indexFM, 
                  uint dollarPositionBWT, uint dollarBaseBWT, uint modposdollarBWT, 
                  uint numQueries, uint sizeQuery, unsigned char *queries, uint *results)
{ 
  unsigned int *groupQueries = NULL;
  uint i, j;
  
  uint indexBase, flg2, flg3;
  uint word, queryWord, bitCount;
  uint bit0, bit1;
  uint nQueryWords = sizeQuery/4;
    
  uint globalThreadIdx, localThreadIdx, localWarpIdx, globalWarpIdx, intraRequestThreadIdx;

  __shared__ s_interval_t   s_posIndexFM[NUM_WARPS_PER_BLOCK]; 
  __shared__ s_entry_t      s_indexFM[NUM_WARPS_PER_BLOCK]; 

  globalThreadIdx = blockIdx.x * MAX_THREADS_PER_SM + threadIdx.x;

  if ((threadIdx.x < MAX_THREADS_PER_SM) && (globalThreadIdx < (numQueries * 2))){

    globalWarpIdx         = globalThreadIdx / SIZE_WARP;
    localWarpIdx          = threadIdx.x / SIZE_WARP;
    localThreadIdx        = threadIdx.x % SIZE_WARP;
    intraRequestThreadIdx = localThreadIdx % THREADS_PER_REQUEST;

    uint interval = (localThreadIdx % 2) ? bwtsize : 0;
    groupQueries = (unsigned int *) (queries + ((globalWarpIdx / 2) * SIZE_WARP * sizeQuery));
    groupQueries = (globalWarpIdx % 2) ? groupQueries + 16 : groupQueries;

    for(queryWord = 0; queryWord < nQueryWords; queryWord++){
      word = groupQueries[queryWord * SIZE_WARP + (localThreadIdx / 2)] >> 1;
      // Perform 4 LF-mappings
      #pragma unroll 1
      for(j = 0; j < 4/K_STEPS; j++) {
        // Decoding bases
        bit1 = word & 0x02000000;
        flg2 = word & 0x01000000;
        flg3 = flg2 ^ 0x01000000;
        bit0 = bit1 ? flg3 : flg2;
        indexBase = (bit1 | bit0) >> 24;
        word <<= 8;
        // Searching both intervals in parallel (L or R)
        s_posIndexFM[localWarpIdx].interval[localThreadIdx] = interval / NUM_CHUNK;
        #pragma unroll 1
        for(i=0; i<NUM_LOADS; i+=1){
          uint nSharedEntry0 = ((REQUESTS_PER_WARP * i) + (localThreadIdx / THREADS_PER_REQUEST));
          uint4 loadData0 = ((uint4 *) (indexFM[s_posIndexFM[localWarpIdx].interval[nSharedEntry0]].data))[intraRequestThreadIdx];
          s_indexFM[localWarpIdx].entryV4[nSharedEntry0][intraRequestThreadIdx] = loadData0;
        }
        // Processing the interval (L or R)
        bitCount = countBitmaps(interval, localWarpIdx, localThreadIdx, bit0, bit1, s_indexFM);
        if(modposdollarBWT==(interval / NUM_CHUNK))
          bitCount = ((indexBase == dollarBaseBWT) && (interval > dollarPositionBWT)) ? bitCount-1 : bitCount;
        interval = s_indexFM[localWarpIdx].entry[localThreadIdx][TOTAL_NUM_BITMAPS + indexBase] + bitCount;
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


