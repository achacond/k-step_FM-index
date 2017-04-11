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
#include "omp.h"

#define   CHUNK_QUERIES 16

typedef struct {
  uint bitmap[BITS_PER_BASE * NUM_BITMAPS * K_STEPS];
  uint cnt[NUM_COUNTERS];
} bitcnt_t;

size_t sizeOfEntry(){
  return(sizeof(bitcnt_t));
}

void searchIndex(void *index, void *dataqueries, void *resIntervals) 
{  
  fmi_t *fmi=(fmi_t *) index;
  res_t *res=(res_t *) resIntervals;
  qrys_t *qrys=(qrys_t *) dataqueries;

  bitcnt_t *indexFM = (bitcnt_t *)fmi->h_index;
  uint bwtsize = fmi->bwtsize;

  uint numqueries = qrys->num;
  uint qrysize = qrys->size;
  unsigned char *queries = qrys->h_queries;
  uint *results = res->h_results;

  unsigned char *query = NULL;
  uint L, R;
  uint iterations = qrysize - 1, cntquery, indexresults; 
  int j, i, n;
  
  uint bcountL, bcountR, indexBase, aux_bit1, aux_bit0, flg2, flg3;
  uint base, bitCount, indexCounterL, indexCounterR, bitmap=0xFFFFFFFF, bitmapShifted, mask = 0xFFFFFFFF;
  uint bitmap1, bitmap0, not_bitmap0, not_bitmap1;
  int shift;

  uint dollarPositionBWT[K_STEPS];
  uint dollarBaseBWT[K_STEPS];
  uint modposdollarBWT[K_STEPS];

  uint bases[CHUNK_QUERIES][K_STEPS];
  uint bit0[CHUNK_QUERIES][K_STEPS];
  uint bit1[CHUNK_QUERIES][K_STEPS];

  // Virtualizing by software the SMT pasallelism
  uint vthread;
  uint vectorL[CHUNK_QUERIES];
  uint vectorR[CHUNK_QUERIES];
  char *vectorQ[CHUNK_QUERIES];
  uint vectorIndexBase[CHUNK_QUERIES];

  for(i=0; i<K_STEPS; i++){
    dollarPositionBWT[i] = fmi->h_dollarPositionBWT[i];
    dollarBaseBWT[i] = fmi->h_dollarBaseBWT[i];
    modposdollarBWT[i] = fmi->h_dollarPositionBWT[i] / NUM_CHUNK;
  }
  
  #pragma omp for schedule (static)
  for (cntquery = 0; cntquery < numqueries; cntquery += CHUNK_QUERIES) {  

    for(vthread = 0; vthread < CHUNK_QUERIES; vthread++){
      vectorL[vthread] = 0;      
      vectorR[vthread] = bwtsize;
      vectorQ[vthread] = queries + ((cntquery + vthread) * qrysize);
    }

    for(j = iterations; j >= 0; j -= K_STEPS)
    {
      for(vthread = 0; vthread < CHUNK_QUERIES; vthread++){
        for(i = K_STEPS - 1; i >= 0; i--){
          query = vectorQ[vthread];
          bases[vthread][i] = (uint) query[j-i];
        }
      }
      
      for(vthread = 0; vthread < CHUNK_QUERIES; vthread++)
        for(i = 0; i < N_CACHE_LINES; i++)
          _mm_prefetch(((char *) indexFM[vectorL[vthread] / NUM_CHUNK].bitmap) + (i * SIZE_CACHE_LINE), _MM_HINT_NTA);
      
      for(vthread = 0; vthread < CHUNK_QUERIES; vthread++)
        for(i = 0; i < N_CACHE_LINES; i++)
          _mm_prefetch(((char *) indexFM[vectorR[vthread] / NUM_CHUNK].bitmap) + (i * SIZE_CACHE_LINE), _MM_HINT_NTA);
  
      for(vthread = 0; vthread < CHUNK_QUERIES; vthread++){
        vectorIndexBase[vthread] = 0x0;
        for(i=0; i<K_STEPS; i++){
          base = bases[vthread][i];
          aux_bit1 = base & 0x04;
          flg2 = base & 0x02;
          flg3 = flg2 ^ 0x02;
          aux_bit0 = aux_bit1 ? flg3 : flg2;
  
          bit0[vthread][i] = aux_bit0;
          bit1[vthread][i] = aux_bit1;
      
          vectorIndexBase[vthread] |= (aux_bit1 | aux_bit0) << (i*2);
        }
        vectorIndexBase[vthread] >>= 1;
      }

      for(vthread = 0; vthread < CHUNK_QUERIES; vthread++){

        L = vectorL[vthread];
        R = vectorR[vthread];
        indexCounterL = L / NUM_CHUNK;
        indexCounterR = R / NUM_CHUNK;
        indexBase = vectorIndexBase[vthread];

        bcountL = indexFM[indexCounterL].cnt[indexBase];
        bcountR = indexFM[indexCounterR].cnt[indexBase];

        // Searching the L interval
        shift = L % NUM_CHUNK;
        bitCount = 0;
        
        for(n = 0; n < NUM_BITMAPS; n++){
          bitmapShifted = mask << (32 - shift);
          bitmap = (shift > 32) ? mask : bitmapShifted;
          bitmap = (shift > 0) ? bitmap : 0x0;
      
          for(i = 0; i < K_STEPS; i++){
            bitmap0 = indexFM[indexCounterL].bitmap[(2 * NUM_BITMAPS * i) + n];
            bitmap1 = indexFM[indexCounterL].bitmap[(2 * NUM_BITMAPS * i) + NUM_BITMAPS + n];
            not_bitmap0 = ~bitmap0;
            not_bitmap1 = ~bitmap1;
            bitmap0 = bit0[vthread][i] ? bitmap0 : not_bitmap0;
            bitmap1 = bit1[vthread][i] ? bitmap1 : not_bitmap1;
            bitmap &= (bitmap0 & bitmap1); 
          }
          bitCount += _mm_popcnt_u32(bitmap);
          shift -= 32;
        }

        for(i = 0; i < K_STEPS; i++){
          if(modposdollarBWT[i] == indexCounterL)
            bitCount = ((indexBase == dollarBaseBWT[i]) && (L > dollarPositionBWT[i])) ? bitCount - 1 : bitCount;
        }      
        L = bcountL + bitCount;
        
        // Searching the R interval
        shift = R % NUM_CHUNK;
        bitCount = 0;
        
        for(n = 0; n < NUM_BITMAPS; n++){
          bitmapShifted = mask << (32-shift);
          bitmap = (shift > 32) ? mask : bitmapShifted;
          bitmap = (shift > 0) ? bitmap : 0x0;
      
          for(i=0; i<K_STEPS; i++){
            bitmap0 = indexFM[indexCounterR].bitmap[(2 * NUM_BITMAPS * i) + n];
            bitmap1 = indexFM[indexCounterR].bitmap[(2 * NUM_BITMAPS * i) + NUM_BITMAPS + n];
            not_bitmap0 = ~bitmap0;
            not_bitmap1 = ~bitmap1;
            bitmap0 = bit0[vthread][i] ? bitmap0 : not_bitmap0;
            bitmap1 = bit1[vthread][i] ? bitmap1 : not_bitmap1;
            bitmap &= (bitmap0 & bitmap1); 
          }
          bitCount += _mm_popcnt_u32(bitmap);
          shift -= 32;
        }

        for(i = 0; i < K_STEPS; i++){
          if(modposdollarBWT[i] == indexCounterR)
            bitCount = ((indexBase == dollarBaseBWT[i]) && (R > dollarPositionBWT[i])) ? bitCount - 1 : bitCount;
        }      

        R = bcountR + bitCount;

        vectorL[vthread] = L;
        vectorR[vthread] = R;
      }
    }

    for(vthread = 0; vthread < CHUNK_QUERIES; vthread++){
      results[(cntquery + vthread)  * 2]      = vectorL[vthread];
      results[((cntquery + vthread) * 2) + 1] = vectorR[vthread];
    }
  }
}

