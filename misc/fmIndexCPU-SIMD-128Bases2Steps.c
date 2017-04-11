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
 */s

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/common.h"
#include "../common/interface.h"
#include <nmmintrin.h>

typedef struct {
  uint bitmap[16];
  uint cnt[16];
} bitcnt_t;

typedef struct {
  uint steps;
  uint bwtsize;
  uint ncounters;
  uint nentries;
  uint chunk;
  uint nbitmaps;
  uint dollarPositionBWT[2];
  uint dollarBaseBWT[2];
    bitcnt_t *index;
} fmi_t;

int loadIndex(const char *fn, void **index)
{
  FILE *fp=NULL;
  fmi_t *fmi = (fmi_t*)malloc(sizeof(fmi_t));
  size_t result;
  int i;

  fp=fopen(fn, "rb");
  if (fp==NULL) return (E_OPENING_INDEX_FILE);

  fread(&fmi->steps, sizeof(uint), 1, fp);
  fread(&fmi->bwtsize, sizeof(uint), 1, fp);
  fread(&fmi->ncounters, sizeof(uint), 1, fp);
  fread(&fmi->nentries, sizeof(uint), 1, fp);
  fread(&fmi->chunk, sizeof(uint), 1, fp);

  for(i=0; i<fmi->steps; i++)
    fread(&fmi->dollarPositionBWT[i], sizeof(uint), 1, fp);
  for(i=0; i<fmi->steps; i++)
    fread(&fmi->dollarBaseBWT[i], sizeof(uint), 1, fp);

  //SIMD using 16Bytes aligned memory: (bitcnt_t*) memalign(16, (fmi->ncounters)*sizeof(bitcnt_t));
  fmi->index = (bitcnt_t*)malloc((fmi->nentries)*sizeof(bitcnt_t));
  if (fmi->index==NULL) return (E_ALLOCATING_FMI_COUNTERS);

  result=fread(fmi->index, sizeof(bitcnt_t), (fmi->nentries), fp);
  if (result != fmi->nentries) return (E_READING_FMI);

  fclose(fp);
  (*index)=fmi;
  return (SUCCESS);
}

void searchIndexCPU(void *index, void *dataqueries, void *resIntervals) 
{  
  fmi_t *fmi=(fmi_t *) index;
  res_t *res=(res_t *) resIntervals;
  qrys_t *qrys=(qrys_t *) dataqueries;

  bitcnt_t *indexFM = fmi->index;

  uint bwtsize = fmi->bwtsize;
  uint chunk = fmi->chunk;

  uint numqueries = qrys->num;
  uint qrysize = qrys->size;
  char *queries = qrys->queries;
  uint *results = res->results;

  unsigned short *query = NULL; 
  unsigned char indexCnt;
  uint L, R;
  uint iterations = (qrysize/2) - 1, cntquery, indexresults;
  int j, i;

  uint B1_indexCnt, B1_bit1, B1_bit0, B1_flg2, B1_flg3;
  uint B0_indexCnt, B0_bit1, B0_bit0, B0_flg2, B0_flg3;
  uint base, bitCount, counter, indexCounter;
  unsigned short shift;

  __m128i *pindex;
  typedef union { __m128i i128; unsigned int u32[4]; unsigned long long u64[2]; } vect4uint;
  vect4uint btmp0, btmp1, btmp2, btmp3, bitmap, auxbitmap, mask, vectorIndexCnt;
  mask.i128 = _mm_set1_epi32(0xFFFFFFFF);
  uint matrixMask[516] = MATRIXMASK128;
  __m128i *pmatrixMask = (__m128i*) matrixMask;

  #pragma omp for schedule (static)    
  for (cntquery = 0; cntquery<numqueries; cntquery++) {    
    L = 0; R = bwtsize;

    query = (unsigned short *) (queries+(cntquery*qrysize));

    for(j=iterations; j>=0; j--) {
      // Getting 2 bases
      base = (uint) query[j];
      //Switch order of bases and extract masks
      //BYTE 1(little-endian)          //BYTE 0(little-endian)
      B1_bit1 = base & 0x0004;        B0_bit1 = base & 0x0400;
      B1_flg2 = base & 0x0002;        B0_flg2 = base & 0x0200;
      B1_flg3 = B1_flg2 ^ 0x0002;      B0_flg3 = B0_flg2 ^ 0x0200;
      B1_bit0 = B1_bit1 ? B1_flg3 : B1_flg2;  B0_bit0 = B0_bit1 ? B0_flg3 : B0_flg2;  

      B1_indexCnt = B1_bit1 | B1_bit0;    B0_indexCnt = B0_bit1 | B0_bit0;
      B1_indexCnt <<= 1;            B0_indexCnt >>= 9;
  
      indexCnt = B1_indexCnt | B0_indexCnt;
      vectorIndexCnt.i128 = _mm_set1_epi8((unsigned char) indexCnt << 4);
    
      // Searching for L interval
      indexCounter = L >> 7;
      shift = L & 127;
      pindex = (__m128i*) &(indexFM[indexCounter].bitmap);
      
      btmp0.i128 = _mm_load_si128(pindex);
      btmp1.i128 = _mm_load_si128(pindex+1);
      btmp2.i128 = _mm_load_si128(pindex+2);
      btmp3.i128 = _mm_load_si128(pindex+3);

      btmp3.i128 = _mm_blendv_epi8 (_mm_andnot_si128(btmp3.i128, mask.i128), btmp3.i128, vectorIndexCnt.i128);
      btmp2.i128 = _mm_blendv_epi8 (_mm_andnot_si128(btmp2.i128, mask.i128), btmp2.i128, _mm_slli_epi32(vectorIndexCnt.i128, 1));
      btmp1.i128 = _mm_blendv_epi8 (_mm_andnot_si128(btmp1.i128, mask.i128), btmp1.i128, _mm_slli_epi32(vectorIndexCnt.i128, 2));
      btmp0.i128 = _mm_blendv_epi8 (_mm_andnot_si128(btmp0.i128, mask.i128), btmp0.i128, _mm_slli_epi32(vectorIndexCnt.i128, 3));
      
      bitmap.i128 = _mm_and_si128(_mm_load_si128(pmatrixMask+shift), _mm_and_si128(_mm_and_si128(btmp0.i128, btmp1.i128), _mm_and_si128(btmp2.i128, btmp3.i128)));
      counter = (uint) _mm_popcnt_u64(bitmap.u64[0]) + (uint) _mm_popcnt_u64(bitmap.u64[1]);
      L = indexFM[indexCounter].cnt[indexCnt] + counter;


      // Searching for R interval
      indexCounter = R >> 7;
      shift = R & 127;
      pindex = (__m128i*) &(indexFM[indexCounter].bitmap);
  
      btmp0.i128 = _mm_load_si128(pindex);
      btmp1.i128 = _mm_load_si128(pindex+1);
      btmp2.i128 = _mm_load_si128(pindex+2);
      btmp3.i128 = _mm_load_si128(pindex+3);

      btmp3.i128 = _mm_blendv_epi8 (_mm_andnot_si128(btmp3.i128, mask.i128), btmp3.i128, vectorIndexCnt.i128);
      btmp2.i128 = _mm_blendv_epi8 (_mm_andnot_si128(btmp2.i128, mask.i128), btmp2.i128, _mm_slli_epi32(vectorIndexCnt.i128, 1));
      btmp1.i128 = _mm_blendv_epi8 (_mm_andnot_si128(btmp1.i128, mask.i128), btmp1.i128, _mm_slli_epi32(vectorIndexCnt.i128, 2));
      btmp0.i128 = _mm_blendv_epi8 (_mm_andnot_si128(btmp0.i128, mask.i128), btmp0.i128, _mm_slli_epi32(vectorIndexCnt.i128, 3));
  
      bitmap.i128 = _mm_and_si128(_mm_load_si128(pmatrixMask+shift), _mm_and_si128(_mm_and_si128(btmp0.i128, btmp1.i128), _mm_and_si128(btmp2.i128, btmp3.i128)));
      counter = (uint) _mm_popcnt_u64(bitmap.u64[0]) + (uint) _mm_popcnt_u64(bitmap.u64[1]);
      R = indexFM[indexCounter].cnt[indexCnt] + counter;
    }

    indexresults = cntquery << 1;
    results[indexresults] = L;
    results[indexresults+1] = R;
  }
}

int saveResults(const char *fn, void *results, void *index)
{
  fmi_t *fmi=(fmi_t *) index;
  res_t *res=(res_t *) results;

  char resultsFileOut[512];
  int error;

  #ifdef CUDA
    sprintf(resultsFileOut, "%sres.gpu", fn);
  #else
    sprintf(resultsFileOut, "%sres.cpu", fn);
  #endif

  error=writeResults(resultsFileOut, res->results, res->num);
  HOST_HANDLE_ERROR(error);

  return (SUCCESS);
}

int freeIndex(void **index) 
{  
  fmi_t *fmi = (fmi_t *) (*index);

  if(fmi->index != NULL){
    free(fmi->index);
    fmi->index=NULL;
  }

  return(SUCCESS);
}

int freeReference(void **reference, void **index) 
{  
  ref_t *ref = (ref_t *) (*reference);
  
  if(ref->reference != NULL)
    free(ref->reference);

  ref->reference=NULL;
  return(SUCCESS);
}

