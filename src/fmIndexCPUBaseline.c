/*
 *  k-step FM-index (benchmarking for CPU and GPU)
 *  Copyright (c) 2013-2017 by Alejandro Chacon  <alejandro.chacond@gmail.com>
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
#include "../common/interface.h"
#include <nmmintrin.h>
#include "omp.h"

#if defined(K_STEPS) || defined(NUM_BITMAPS) || defined(NUM_COUNTERS)
#else
  #define K_STEPS         1
  #define NUM_CHUNK       64
  #define NUM_COUNTERS    4
#endif

#if defined(INTERLEAVE_BMP)
  #define IN_INDEX_TAG      101
#else
  #define IN_INDEX_TAG      100
#endif

#define SIZE_ALPHABET     4
#define NUM_BITMAPS       NUM_CHUNK/32
#define BITS_PER_BASE     2
#define SIZE_CACHE_LINE   64
#define N_CACHE_LINES     (BITS_PER_BASE*NUM_BITMAPS*K_STEPS)*4/SIZE_CACHE_LINE

typedef struct {
  uint32_t bitmap[BITS_PER_BASE*NUM_BITMAPS*K_STEPS];
  uint32_t cnt[NUM_COUNTERS];
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

int32_t loadIndex(const char *fn, void **index)
{
  FILE *fp=NULL;
  fmi_t *fmi = (fmi_t*)malloc(sizeof(fmi_t));
  uint32_t index_tag = 0;
  size_t result;
  int32_t i;

  fp = fopen(fn, "rb");
  if (fp == NULL) return (E_OPENING_INDEX_FILE);

  result = fread(&index_tag, sizeof(uint32_t), 1, fp);
  if (result != 1) return (E_READING_FMI);
  printf("Index Version: %u - Require: %u\n", index_tag, IN_INDEX_TAG);

  result = fread(&fmi->steps, sizeof(uint32_t), 1, fp);
  if (result != 1) return (E_READING_FMI);
  printf("Steps (k): %u \n", fmi->steps);

  result = fread(&fmi->bwtsize, sizeof(uint32_t), 1, fp);
  if (result != 1) return (E_READING_FMI);
  printf("Reference Size: %u \n", fmi->bwtsize - 1);

  result = fread(&fmi->ncounters, sizeof(uint32_t), 1, fp);
  if (result != 1) return (E_READING_FMI);
  printf("rLF counters: %u \n", fmi->ncounters);

  result = fread(&fmi->nentries, sizeof(uint32_t), 1, fp);
  if (result != 1) return (E_READING_FMI);
  printf("F entries: %u \n", fmi->nentries);

  result = fread(&fmi->chunk, sizeof(uint32_t), 1, fp);
  if (result != 1) return (E_READING_FMI);
  printf("d Sampling: %u \n", fmi->chunk);

  if((index_tag == IN_INDEX_TAG) && (fmi->steps == K_STEPS) && 
     (fmi->ncounters == NUM_COUNTERS) && (fmi->chunk == NUM_CHUNK)){

    fmi->h_dollarPositionBWT = (uint32_t*)malloc((fmi->steps)*sizeof(uint32_t));
    if (fmi->h_dollarPositionBWT==NULL) return (E_ALLOCATING_FMI);
    result = fread(fmi->h_dollarPositionBWT, sizeof(uint32_t), fmi->steps, fp);
    if (result != fmi->steps) return (E_READING_FMI);

    fmi->h_dollarBaseBWT = (uint32_t*)malloc((fmi->steps)*sizeof(uint32_t));
    if (fmi->h_dollarBaseBWT==NULL) return (E_ALLOCATING_FMI);
    result = fread(fmi->h_dollarBaseBWT, sizeof(uint32_t), fmi->steps, fp);
    if (result != fmi->steps) return (E_READING_FMI);

    fmi->h_modposdollarBWT = (uint32_t*)malloc((fmi->steps)*sizeof(uint32_t));
    if (fmi->h_modposdollarBWT==NULL) return (E_ALLOCATING_FMI);
    for(i=0; i<fmi->steps; i++)
    fmi->h_modposdollarBWT[i]=fmi->h_dollarPositionBWT[i]/NUM_CHUNK;

    fmi->h_index = (bitcnt_t*)malloc((fmi->nentries)*sizeof(bitcnt_t));
    if (fmi->h_index==NULL) return (E_ALLOCATING_FMI);

    result = fread(fmi->h_index, sizeof(bitcnt_t), fmi->nentries, fp);
    if (result != fmi->nentries) return (E_READING_FMI);

    fmi->d_dollarPositionBWT = NULL;
    fmi->d_dollarBaseBWT = NULL;
    fmi->d_modposdollarBWT = NULL;
    fmi->d_index = NULL;

    fclose(fp);
    (*index)=fmi;
    return (SUCCESS);
  }else{
    fclose(fp);
    free(fmi);
    return(IN_INDEX_TAG);
  }
}

int32_t freeIndex(void **index)
{
  fmi_t *fmi = (fmi_t *) (*index);

  if(fmi->h_index != NULL){
    free(fmi->h_index);
    fmi->h_index=NULL;
  }

  return(SUCCESS);
}

void searchIndexCPU(void *index, void *dataqueries, void *resIntervals)
{
  fmi_t *fmi=(fmi_t *) index;
  res_t *res=(res_t *) resIntervals;
  qrys_t *qrys=(qrys_t *) dataqueries;

  bitcnt_t *indexFM = fmi->h_index;
  uint32_t bwtsize = fmi->bwtsize;

  uint32_t numqueries = qrys->num;
  uint32_t qrysize = qrys->size;
  char     *queries = qrys->h_queries;
  uint32_t *results = res->h_results;

  char     *query = NULL;
  uint32_t L, R;
  uint32_t iterations = qrysize - 1, cntquery, indexresults;
  int32_t  j, i, n;

  uint32_t bcountL, bcountR, indexBase, aux_bit1, aux_bit0, flg2, flg3;
  uint32_t base, indexCounterL, indexCounterR, bitmap = 0xFFFFFFFF, bitmapShifted, mask = 0xFFFFFFFF;
  uint32_t bitmap1, bitmap0, not_bitmap0, not_bitmap1;
  int32_t  shift, bitCount;

  uint32_t dollarPositionBWT[K_STEPS];
  uint32_t dollarBaseBWT[K_STEPS];
  uint32_t modposdollarBWT[K_STEPS];

  uint32_t bases[K_STEPS];
  uint32_t bit0[K_STEPS];
  uint32_t bit1[K_STEPS];

  for(i=0; i<K_STEPS; i++){
    dollarPositionBWT[i] = fmi->h_dollarPositionBWT[i];
    dollarBaseBWT[i] = fmi->h_dollarBaseBWT[i];
    modposdollarBWT[i] = fmi->h_dollarPositionBWT[i] / NUM_CHUNK;
  }

  #pragma omp for schedule (static)
  for (cntquery = 0; cntquery<numqueries; cntquery++) {
    L = 0; R = bwtsize;
    query = queries+(cntquery*qrysize);

    for(j=iterations; j>=0; j-=K_STEPS) {

      for(i=K_STEPS-1; i>=0; i--)
        bases[i] = (uint32_t) query[j-i];

      indexCounterL = L / NUM_CHUNK;
      for(i=0; i<N_CACHE_LINES; i++)
        _mm_prefetch(((char *)indexFM[indexCounterL].bitmap)+(i*SIZE_CACHE_LINE), _MM_HINT_NTA);

      indexCounterR = R / NUM_CHUNK;
      for(i=0; i<N_CACHE_LINES; i++)
        _mm_prefetch(((char *)indexFM[indexCounterR].bitmap)+(i*SIZE_CACHE_LINE), _MM_HINT_NTA);

      indexBase=0x0;
      for(i=0; i<K_STEPS; i++){
        base = bases[i];
        aux_bit1 = base & 0x04;
        flg2 = base & 0x02;
        flg3 = flg2 ^ 0x02;
        aux_bit0 = aux_bit1 ? flg3 : flg2;

        bit0[i] = aux_bit0;
        bit1[i] = aux_bit1;

        indexBase |= (aux_bit1 | aux_bit0) << (i*2);
      }
      indexBase >>= 1;
      bcountL = indexFM[indexCounterL].cnt[indexBase];
      bcountR = indexFM[indexCounterR].cnt[indexBase];

      // Searching L interval
      shift = L % NUM_CHUNK;
      bitCount = 0;

      for(n=0; n<NUM_BITMAPS; n++){
        bitmapShifted = mask << (32-shift);
        bitmap = (shift>32) ? mask : bitmapShifted;
        bitmap = (shift>0) ? bitmap : 0x0;

        for(i=0; i<K_STEPS; i++){
          bitmap0 = indexFM[indexCounterL].bitmap[(2*NUM_BITMAPS*i)+n];
          bitmap1 = indexFM[indexCounterL].bitmap[(2*NUM_BITMAPS*i)+NUM_BITMAPS+n];
          not_bitmap0 = ~bitmap0;
          not_bitmap1 = ~bitmap1;
          bitmap0 = bit0[i] ? bitmap0 : not_bitmap0;
          bitmap1 = bit1[i] ? bitmap1 : not_bitmap1;
          bitmap &= (bitmap0 & bitmap1);
        }
        bitCount += _mm_popcnt_u32(bitmap);
        shift -= 32;
      }

      for(i=0; i<K_STEPS; i++){
        if((modposdollarBWT[i]==indexCounterL) && (indexBase == dollarBaseBWT[i]) && (L>dollarPositionBWT[i])){
          bitCount--;
        }
      }
      L = bcountL + bitCount;

      // Searching R interval
      shift = R % NUM_CHUNK;
      bitCount = 0;

      for(n=0; n<NUM_BITMAPS; n++){
        bitmapShifted = mask << (32-shift);
        bitmap = (shift>32) ? mask : bitmapShifted;
        bitmap = (shift>0) ? bitmap : 0x0;

        for(i=0; i<K_STEPS; i++){
          bitmap0 = indexFM[indexCounterR].bitmap[(2*NUM_BITMAPS*i)+n];
          bitmap1 = indexFM[indexCounterR].bitmap[(2*NUM_BITMAPS*i)+NUM_BITMAPS+n];
          not_bitmap0 = ~bitmap0;
          not_bitmap1 = ~bitmap1;
          bitmap0 = bit0[i] ? bitmap0 : not_bitmap0;
          bitmap1 = bit1[i] ? bitmap1 : not_bitmap1;
          bitmap &= (bitmap0 & bitmap1);
        }
        bitCount += _mm_popcnt_u32(bitmap);
        shift -= 32;
      }

      for(i=0; i<K_STEPS; i++){
        if((modposdollarBWT[i]==indexCounterR) && (indexBase == dollarBaseBWT[i]) && (R>dollarPositionBWT[i])){
          bitCount--;
        }
      }
      R = bcountR + bitCount;
    }
    indexresults = cntquery << 1;
    results[indexresults] = L;
    results[indexresults+1] = R;
  }
}
