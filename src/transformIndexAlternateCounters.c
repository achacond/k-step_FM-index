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
#include <nmmintrin.h>
#include <stdint.h>
#include "../common/common.h"

#if defined(K_STEPS) || defined(NUM_BITMAPS) || defined(NUM_COUNTERS)
#else
  #define K_STEPS      1
  #define NUM_CHUNK    64
  #define NUM_COUNTERS 4
#endif

#define OUT_INDEX_CPU_TAG   200
#define OUT_INDEX_GPU_TAG   201
#define IN_INDEX_TAG        100
#define NUM_COUNTERS_SLIM   ( NUM_COUNTERS / 2 )
#define NUM_BITMAPS         ( NUM_CHUNK / 32 )
#define BITS_PER_BASE       2
#define INTERLEAVING_FMI    4

typedef struct {
  uint32_t bitmap[BITS_PER_BASE * NUM_BITMAPS * K_STEPS];
  uint32_t cnt[NUM_COUNTERS];
} bitcnt_t;

typedef struct {
  uint32_t cnt[NUM_COUNTERS_SLIM];
  uint32_t bitmap[BITS_PER_BASE * NUM_BITMAPS * K_STEPS];
} newbitcnt_t;

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
  newbitcnt_t *h_index;
  uint32_t *d_dollarPositionBWT;
  uint32_t *d_dollarBaseBWT;
  uint32_t *d_modposdollarBWT;
  newbitcnt_t *d_index;
} newfmi_t;

int32_t mod(int32_t x, int32_t m) {
  return (x%m + m) % m;
}

uint32_t countEntry(void *index, uint32_t idxEntry, uint32_t letter, int32_t position)
{
  fmi_t *fmi = (fmi_t *) index;

  uint32_t bit0[fmi->steps];
  uint32_t bit1[fmi->steps];
  uint32_t bitCount = 0;
  uint32_t i, n;
  uint32_t bitmapShifted, mask = 0xFFFFFFFF, bitmap = 0xFFFFFFFF;
  uint32_t bitmap1, bitmap0, not_bitmap0, not_bitmap1;
  
  for(i = 0; i < fmi->steps; i++){
    bit0[i] = letter & 0x1;
    bit1[i] = (letter >> 1) & 0x1;
    letter >>= 2;
  }

  for(n = 0; n < NUM_BITMAPS; n++){
    bitmapShifted = mask << (32 - position);
    bitmap = (position > 32) ? mask : bitmapShifted;
    bitmap = (position > 0) ? bitmap : 0x0;

    for(i = 0; i < fmi->steps; i++){
      bitmap0 = fmi->h_index[idxEntry].bitmap[(2 * NUM_BITMAPS * i) + n];
      bitmap1 = fmi->h_index[idxEntry].bitmap[(2 * NUM_BITMAPS * i) + NUM_BITMAPS + n];
      not_bitmap0 = ~bitmap0;
      not_bitmap1 = ~bitmap1;
      bitmap0 = bit0[i] ? bitmap0 : not_bitmap0;
      bitmap1 = bit1[i] ? bitmap1 : not_bitmap1;
      bitmap &= (bitmap0 & bitmap1);
    }
    bitCount += _mm_popcnt_u32(bitmap);
    position -= 32;
  }

  return(bitCount);
}

int32_t initIndex(void *index, void **newIndex)
{
  newfmi_t *newfmi = (newfmi_t *) malloc(sizeof(newfmi_t));
  fmi_t *fmi = (fmi_t *) index;
  int32_t i;

  newfmi->steps      = fmi->steps;
  newfmi->bwtsize    = fmi->bwtsize;
  newfmi->ncounters  = fmi->ncounters / 2;
  newfmi->nentries   = fmi->nentries + 1;
  newfmi->chunk      = fmi->chunk;

  newfmi->h_dollarPositionBWT = (uint32_t*)malloc((newfmi->steps) * sizeof(uint32_t));
  if (newfmi->h_dollarPositionBWT == NULL) return (E_ALLOCATING_FMI);
  newfmi->h_dollarBaseBWT = (uint32_t*)malloc((newfmi->steps) * sizeof(uint32_t));
  if (newfmi->h_dollarBaseBWT == NULL) return (E_ALLOCATING_FMI);

  newfmi->h_index = (newbitcnt_t*)malloc((newfmi->nentries) * sizeof(newbitcnt_t));
  if (newfmi->h_index==NULL) return (E_ALLOCATING_FMI);

  for(i=0; i<fmi->steps; i++){
    newfmi->h_dollarPositionBWT[i] = fmi->h_dollarPositionBWT[i];
    newfmi->h_dollarBaseBWT[i]     = fmi->h_dollarBaseBWT[i];
  }

  newfmi->d_dollarPositionBWT = NULL;
  newfmi->d_dollarBaseBWT     = NULL;
  newfmi->d_index             = NULL;

  (*newIndex) = newfmi;
  return (SUCCESS);

}

int32_t saveIndexCPU(char *fn, void *index)
{
  newfmi_t *newfmi = (newfmi_t *) index;
  uint32_t i, error, id_index = OUT_INDEX_CPU_TAG;
  char fmiFileOut[512];
  FILE *fp = NULL;

  sprintf(fmiFileOut, "%s.ac", fn);

  fp = fopen(fmiFileOut, "wb");
  if (fp == NULL) return (E_SAVING_INDEX_FILE);

  fwrite(&id_index, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->steps, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->bwtsize, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->ncounters, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->nentries, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->chunk, sizeof(uint32_t), 1, fp);
  for(i=0; i<newfmi->steps; i++)
    fwrite(&newfmi->h_dollarPositionBWT[i], sizeof(uint32_t), 1, fp);
  for(i=0; i<newfmi->steps; i++)
    fwrite(&newfmi->h_dollarBaseBWT[i], sizeof(uint32_t), 1, fp);

  fwrite(newfmi->h_index, sizeof(newbitcnt_t), newfmi->nentries, fp);
  fclose(fp);
  return (SUCCESS);
}

int32_t saveIndexGPU(char *fn, void *index)
{
  newfmi_t *newfmi = (newfmi_t *) index;
  uint32_t i, error, id_index = OUT_INDEX_GPU_TAG;
  char fmiFileOut[512];
  FILE *fp = NULL;

  sprintf(fmiFileOut, "%s.interleaving.ac", fn);

  fp = fopen(fmiFileOut, "wb");
  if (fp == NULL) return (E_SAVING_INDEX_FILE);

  fwrite(&id_index, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->steps, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->bwtsize, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->ncounters, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->nentries, sizeof(uint32_t), 1, fp);
  fwrite(&newfmi->chunk, sizeof(uint32_t), 1, fp);
  for(i=0; i<newfmi->steps; i++)
    fwrite(&newfmi->h_dollarPositionBWT[i], sizeof(uint32_t), 1, fp);
  for(i=0; i<newfmi->steps; i++)
    fwrite(&newfmi->h_dollarBaseBWT[i], sizeof(uint32_t), 1, fp);

  fwrite(newfmi->h_index, sizeof(newbitcnt_t), newfmi->nentries, fp);
  fclose(fp);
  return (SUCCESS);
}

int32_t loadIndex(char *fn, void **index)
{
  FILE *fp = NULL;
  fmi_t *fmi = (fmi_t*)malloc(sizeof(fmi_t));
  uint32_t index_tag = 0;
  size_t result;
  int32_t i;

  fp=fopen(fn, "rb");
  if (fp==NULL) return (E_OPENING_INDEX_FILE);

  result = fread(&index_tag, sizeof(uint32_t), 1, fp);
  if (result != 1) return (E_READING_FMI);

  if(index_tag == IN_INDEX_TAG){
	result = fread(&fmi->steps, sizeof(uint32_t), 1, fp);
    if (result != 1) return (E_READING_FMI);
	result = fread(&fmi->bwtsize, sizeof(uint32_t), 1, fp);
    if (result != 1) return (E_READING_FMI);
    result = fread(&fmi->ncounters, sizeof(uint32_t), 1, fp);
    if (result != 1) return (E_READING_FMI);
    result = fread(&fmi->nentries, sizeof(uint32_t), 1, fp);
    if (result != 1) return (E_READING_FMI);
    result = fread(&fmi->chunk, sizeof(uint32_t), 1, fp);
    if (result != 1) return (E_READING_FMI);

    fmi->h_dollarPositionBWT = (uint32_t *) malloc((fmi->steps) * sizeof(uint32_t));
    if (fmi->h_dollarPositionBWT==NULL) return (E_ALLOCATING_FMI);
    result = fread(fmi->h_dollarPositionBWT, sizeof(uint32_t), fmi->steps, fp);
    if (result != fmi->steps) return (E_READING_FMI);

    fmi->h_dollarBaseBWT = (uint32_t *) malloc((fmi->steps)*sizeof(uint32_t));
    if (fmi->h_dollarBaseBWT==NULL) return (E_ALLOCATING_FMI);
    result = fread(fmi->h_dollarBaseBWT, sizeof(uint32_t), fmi->steps, fp);
    if (result != fmi->steps) return (E_READING_FMI);

    fmi->h_modposdollarBWT = (uint32_t *) malloc((fmi->steps)*sizeof(uint32_t));
    if (fmi->h_modposdollarBWT==NULL) return (E_ALLOCATING_FMI);
    for(i=0; i<fmi->steps; i++)
      fmi->h_modposdollarBWT[i]=fmi->h_dollarPositionBWT[i]/NUM_CHUNK;

    fmi->h_index = (bitcnt_t*)malloc((fmi->nentries)*sizeof(bitcnt_t));
    if (fmi->h_index==NULL) return (E_ALLOCATING_FMI);

    result = fread(fmi->h_index, sizeof(bitcnt_t), fmi->nentries, fp);
    if (result != fmi->nentries) return (E_READING_FMI);

    fmi->d_dollarPositionBWT = NULL;
    fmi->d_dollarBaseBWT     = NULL;
    fmi->d_modposdollarBWT   = NULL;
    fmi->d_index             = NULL;

    fclose(fp);
    (*index) = fmi;
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
    fmi->h_index = NULL;
  }
  return(SUCCESS);
}

int32_t freeNewIndex(void **index)
{
  newfmi_t *newfmi = (newfmi_t *) (*index);
  if(newfmi->h_index != NULL){
    free(newfmi->h_index);
    newfmi->h_index = NULL;
  }
  return(SUCCESS);
}

void checksumPrintGPU(void *index, void *newIndex, uint32_t indexEntry, uint32_t numEntries)
{
  fmi_t *fmi    = (fmi_t *) index;
  newfmi_t *newfmi = (newfmi_t *) newIndex;

  uint32_t idx, i, numBitmaps, padding, k;
  uint32_t cntCC[newfmi->ncounters];
  uint32_t diffCC[newfmi->ncounters];

  numBitmaps = BITS_PER_BASE * NUM_BITMAPS * K_STEPS;
  printf("NUM ENTRIES: real: %u saved: %u, size: %u\n", newfmi->bwtsize/fmi->chunk, newfmi->nentries, (uint32_t) sizeof(bitcnt_t));

  for (k = 0; k < newfmi->ncounters; k++) {
    cntCC[k] = 0;
    diffCC[k] = 0;
  }
    
  for (i = 0; i < fmi->nentries; i++){
    if ((i % 2) == 0) padding = 0;
    else padding = newfmi->ncounters;  
    
    for (k = 0; k < newfmi->ncounters; k++) {
      if (newfmi->h_index[i].cnt[k] > newfmi->bwtsize) cntCC[k]++;
      if (newfmi->h_index[i].cnt[k] !=  fmi->h_index[i].cnt[padding + k]){
        printf(" --------------------------------------------\n");
        printf("newEntry: %u, oldEntry: %u\n", newfmi->h_index[i].cnt[k], fmi->h_index[i].cnt[k]);
        printf("idxEntry: %u, idxCounter: %u \n", i, k);
        diffCC[k]++;
      }
    }
  }

  printf("--------------------------------------------- \n");
  printf("LAST ENTRY: INDEX[newfmi->nentries - 1] (NEW LAYOUT)\n");
  printf("BITMAPS: ");
  for (i = 0; i < numBitmaps; i++){
    printf("B[%d]: %u - ", i, newfmi->h_index[newfmi->nentries - 1].bitmap[i]);
  }
  printf("\n");
  printf("COUNTERS: ");
  for (i = 0; i < newfmi->ncounters; i++){
    printf("C[%d]: %u - ", i, newfmi->h_index[newfmi->nentries - 1].cnt[i]);
  }
  printf("--------------------------------------------- \n");

  printf("Vector cntCC: \n");
  for (k = 0; k < newfmi->ncounters; k++) {
    printf ("[%d]: %u \n", k, cntCC[k]);
  }
  printf("\n");

  printf("Vector diffCC: \n");
  for (k = 0; k < newfmi->ncounters; k++) {
    printf ("[%d]: %u \n", k, diffCC[k]);
  }
  printf("\n");  

  for(idx = 0; idx < numEntries; idx++){
    printf("INDEX[%d] (OLD LAYOUT)\n", indexEntry+idx);
    printf("BITMAPS: ");
    for (i = 0; i < numBitmaps; i++){
      printf("B[%d]: %d - ", i, fmi->h_index[indexEntry+idx].bitmap[i]);
    }
    printf("\n");
    printf("COUNTERS: ");
    for (i = 0; i < fmi->ncounters; i++){
      printf("B[%d]: %u - ", i, fmi->h_index[indexEntry+idx].cnt[i]);
    }
    printf("\n");
    printf("\n");

    printf("INDEX[%d] (NEW LAYOUT)\n", indexEntry+idx);
    printf("BITMAPS: ");
    for (i = 0; i < numBitmaps; i++){
      printf("B[%d]: %u - ", i, newfmi->h_index[indexEntry+idx].bitmap[i]);
    }
    printf("\n");

    printf("COUNTERS: ");
    for (i = 0; i < newfmi->ncounters; i++){
      printf("C[%d]: %u - ", i, newfmi->h_index[indexEntry+idx].cnt[i]);
    }
    printf("\n");
  }
}

void transformIndexGPU(void *index, void *newIndex)
{
  fmi_t *fmi    = (fmi_t *) index;
  newfmi_t *newfmi = (newfmi_t *) newIndex;

  uint32_t i, k, s;
  uint32_t lastEntry = newfmi->nentries - 1;
  uint32_t lastCnt[fmi->ncounters];

  for (i = 0; i < lastEntry; i++){ 
    for (k = 0; k < NUM_BITMAPS; k++){
      for (s = 0; s < newfmi->steps; s++){
         newfmi->h_index[i].bitmap[(2 * newfmi->steps * k) + (2 * s)    ] = fmi->h_index[i].bitmap[(2 * s * NUM_BITMAPS) + k];
         newfmi->h_index[i].bitmap[(2 * newfmi->steps * k) + (2 * s) + 1] = fmi->h_index[i].bitmap[(2 * s * NUM_BITMAPS) + NUM_BITMAPS + k];
      }
    }
  }

  //Corner case - last entry (all with 0's)
  for (k = 0; k < NUM_BITMAPS; k++){
    for (s = 0; s < newfmi->steps; s++){
      newfmi->h_index[lastEntry].bitmap[(2 * newfmi->steps * k) + (2 * s)    ] = 0;
      newfmi->h_index[lastEntry].bitmap[(2 * newfmi->steps * k) + (2 * s) + 1] = 0;
    }
  }

  for (i = 0; i < lastEntry; i++){
    for (k = 0; k < newfmi->ncounters; k++){
      if(mod(i, 2) == 0) newfmi->h_index[i].cnt[k] = fmi->h_index[i].cnt[k];
      if(mod(i, 2) == 1) newfmi->h_index[i].cnt[k] = fmi->h_index[i].cnt[newfmi->ncounters + k];
    }
  }

  for (k = 0; k < fmi->ncounters; k++)
    lastCnt[k] = 0;

  lastCnt[0] += NUM_CHUNK - mod(fmi->bwtsize, NUM_CHUNK);
  for (k = 0; k < fmi->ncounters; k++)
    lastCnt[k] += countEntry(fmi, fmi->bwtsize / fmi->chunk, k, (int32_t) (fmi->bwtsize % fmi->chunk));

  //Corner case - last entry (re-count last bitmap)
  for (k = 0; k < newfmi->ncounters; k++){
    if(mod(lastEntry, 2) == 0) newfmi->h_index[lastEntry].cnt[k] = fmi->h_index[lastEntry - 1].cnt[k] + lastCnt[k];
    if(mod(lastEntry, 2) == 1) newfmi->h_index[lastEntry].cnt[k] = fmi->h_index[lastEntry - 1].cnt[newfmi->ncounters + k] + lastCnt[newfmi->ncounters + k];
  }
}

void transformIndexCPU(void *index, void *newIndex)
{
  fmi_t *fmi       = (fmi_t *) index;
  newfmi_t *newfmi = (newfmi_t *) newIndex;

  uint32_t i, k, s;
  uint32_t lastEntry = newfmi->nentries - 1;
  uint32_t lastCnt[fmi->ncounters];

  for (i = 0; i < lastEntry; i++){ 
    for (k = 0; k < NUM_BITMAPS; k++) {
      for (s = 0; s < newfmi->steps; s++){
        newfmi->h_index[i].bitmap[(2 * newfmi->steps * k) + (2 * s)    ] = fmi->h_index[i].bitmap[(2 * newfmi->steps * k) + (2 * s)    ];
        newfmi->h_index[i].bitmap[(2 * newfmi->steps * k) + (2 * s) + 1] = fmi->h_index[i].bitmap[(2 * newfmi->steps * k) + (2 * s) + 1];
      }
    }
  }

  //Corner case - last entry (all with 0's)
  for (k = 0; k < NUM_BITMAPS; k++) {
    for (s = 0; s < newfmi->steps; s++){
      newfmi->h_index[lastEntry].bitmap[(2 * newfmi->steps * k) + (2 * s)    ] = 0;
      newfmi->h_index[lastEntry].bitmap[(2 * newfmi->steps * k) + (2 * s) + 1] = 0;
    }
  }

  for (i = 0; i < lastEntry; i++){
    for (k = 0; k < newfmi->ncounters; k++){
      if(mod(i, 2) == 0) newfmi->h_index[i].cnt[k] = fmi->h_index[i].cnt[k];
      if(mod(i, 2) == 1) newfmi->h_index[i].cnt[k] = fmi->h_index[i].cnt[newfmi->ncounters + k];
    }
  }

  for (k = 0; k < fmi->ncounters; k++)
    lastCnt[k] = 0;

  lastCnt[0] += NUM_CHUNK - mod(fmi->bwtsize, NUM_CHUNK);
  for (k = 0; k < fmi->ncounters; k++)
    lastCnt[k] += countEntry(fmi, fmi->bwtsize / fmi->chunk, k, (int32_t) (fmi->bwtsize % fmi->chunk));

  //Corner case - last entry (re-count last bitmap)
  for (k = 0; k < newfmi->ncounters; k++){
    if(mod(lastEntry, 2) == 0) newfmi->h_index[lastEntry].cnt[k] = fmi->h_index[lastEntry - 1].cnt[k] + lastCnt[k];
    if(mod(lastEntry, 2) == 1) newfmi->h_index[lastEntry].cnt[k] = fmi->h_index[lastEntry - 1].cnt[newfmi->ncounters + k] + lastCnt[newfmi->ncounters + k];
  }
}

int32_t main(int32_t argc, char *argv[])
{
  char *indexFile = argv[1];
  uint32_t  indexEntry, numEntries;

  void    *index = NULL;
  void    *newIndexCPU = NULL;
  void    *newIndexGPU = NULL;
  int32_t error;

  if(argc > 2){
    indexEntry = atoi(argv[2]);
    numEntries = atoi(argv[3]);
  }

  error=loadIndex(indexFile, &index);
  HOST_HANDLE_ERROR(error);

  error=initIndex(index, &newIndexCPU);
  HOST_HANDLE_ERROR(error);

  error=initIndex(index, &newIndexGPU);
  HOST_HANDLE_ERROR(error);

  transformIndexCPU(index, newIndexCPU);
  transformIndexGPU(index, newIndexGPU);

  if(argc > 2)
    checksumPrintGPU(index, newIndexGPU, indexEntry, numEntries);

  error = saveIndexCPU(indexFile, newIndexCPU);
  HOST_HANDLE_ERROR(error);

  error = saveIndexGPU(indexFile, newIndexGPU);
  HOST_HANDLE_ERROR(error);

  error=freeIndex(&index);
  HOST_HANDLE_ERROR(error);

  error=freeNewIndex(&newIndexCPU);
  HOST_HANDLE_ERROR(error);

  error=freeNewIndex(&newIndexGPU);
  HOST_HANDLE_ERROR(error);

  return (SUCCESS);
}
