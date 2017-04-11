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

#if defined(K_STEPS) || defined(NUM_BITMAPS)
#else
  #define K_STEPS         1
  #define NUM_CHUNK       64
#endif

#define OUT_INDEX_TAG     101
#define IN_INDEX_TAG      100
#define NUM_BITMAPS       (NUM_CHUNK / 32)
#define BITS_PER_BASE     2
#define INTERLEAVING_FMI  4

typedef struct {
  uint bitmap[BITS_PER_BASE * NUM_BITMAPS * K_STEPS];
  uint cnt[NUM_COUNTERS];
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

int32_t initIndex(void *index, void **newIndex)
{
  fmi_t *newfmi = (fmi_t *) malloc(sizeof(fmi_t));
  fmi_t *fmi = (fmi_t *) index;
  int32_t i;

  newfmi->steps      = fmi->steps;
  newfmi->bwtsize    = fmi->bwtsize;
  newfmi->ncounters  = fmi->ncounters;
  newfmi->nentries   = fmi->nentries;
  newfmi->chunk      = fmi->chunk;

  newfmi->h_dollarPositionBWT = (uint32_t *) malloc((newfmi->steps) * sizeof(uint32_t));
  if (newfmi->h_dollarPositionBWT == NULL) return (E_ALLOCATING_FMI);
  newfmi->h_dollarBaseBWT = (uint32_t *) malloc((newfmi->steps) * sizeof(uint32_t));
  if (newfmi->h_dollarBaseBWT == NULL) return (E_ALLOCATING_FMI);

  newfmi->h_index = (bitcnt_t *) malloc((newfmi->nentries) * sizeof(bitcnt_t));
  if (newfmi->h_index == NULL) return (E_ALLOCATING_FMI);

  for(i = 0; i < fmi->steps; i++){
    newfmi->h_dollarPositionBWT[i] = fmi->h_dollarPositionBWT[i];
    newfmi->h_dollarBaseBWT[i]     = fmi->h_dollarBaseBWT[i];
  }

  newfmi->d_dollarPositionBWT = NULL;
  newfmi->d_dollarBaseBWT     = NULL;
  newfmi->d_index             = NULL;

  (*newIndex) = newfmi;
  return (SUCCESS);
}

int32_t saveIndex(char *fn, void *index)
{
  fmi_t *fmi = (fmi_t *) index;
  uint32_t i, error, id_index = OUT_INDEX_TAG;
  char fmiFileOut[512];
  FILE *fp=NULL;

  sprintf(fmiFileOut, "%s.interleaving", fn);

  fp = fopen(fmiFileOut, "wb");
  if (fp==NULL) return (E_SAVING_INDEX_FILE);

  fwrite(&id_index, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->steps, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->bwtsize, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->ncounters, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->nentries, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->chunk, sizeof(uint32_t), 1, fp);
  
  for(i = 0; i < fmi->steps; i++)
    fwrite(&fmi->h_dollarPositionBWT[i], sizeof(uint32_t), 1, fp);
  for(i = 0; i < fmi->steps; i++)
    fwrite(&fmi->h_dollarBaseBWT[i], sizeof(uint32_t), 1, fp);

  fwrite(fmi->h_index, sizeof(bitcnt_t), fmi->nentries, fp);
  fclose(fp);
  return (SUCCESS);
}

int32_t loadIndex(char *fn, void **index)
{
  FILE *fp = NULL;
  fmi_t *fmi = (fmi_t*) malloc(sizeof(fmi_t));
  size_t result;
  uint32_t index_tag = 0;
  int32_t i;

  fp = fopen(fn, "rb");
  if (fp == NULL) return (E_OPENING_INDEX_FILE);

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

    fmi->h_dollarPositionBWT = (uint32_t*) malloc((fmi->steps)*sizeof(uint32_t));
    if (fmi->h_dollarPositionBWT == NULL) return (E_ALLOCATING_FMI);
    result = fread(fmi->h_dollarPositionBWT, sizeof(uint32_t), fmi->steps, fp);
    if (result != fmi->steps) return (E_READING_FMI);

    fmi->h_dollarBaseBWT = (uint32_t *) malloc((fmi->steps) * sizeof(uint32_t));
    if (fmi->h_dollarBaseBWT == NULL) return (E_ALLOCATING_FMI);
    result = fread(fmi->h_dollarBaseBWT, sizeof(uint32_t), fmi->steps, fp);
    if (result != fmi->steps) return (E_READING_FMI);

    fmi->h_modposdollarBWT = (uint32_t *) malloc((fmi->steps) * sizeof(uint32_t));
    if (fmi->h_modposdollarBWT == NULL) return (E_ALLOCATING_FMI);
    for(i = 0; i < fmi->steps; i++)
    fmi->h_modposdollarBWT[i] = fmi->h_dollarPositionBWT[i] / NUM_CHUNK;

    fmi->h_index = (bitcnt_t *) malloc((fmi->nentries) * sizeof(bitcnt_t));
    if (fmi->h_index == NULL) return (E_ALLOCATING_FMI);

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

void checksumPrint(void *index, void *newIndex, uint32_t indexEntry, uint32_t numEntries)
{
  fmi_t *fmi    = (fmi_t *) index;
  fmi_t *newfmi = (fmi_t *) newIndex;

  uint32_t idx, i, numBitmaps, numCounters, k;
  uint32_t cntCC[NUM_COUNTERS];
  uint32_t diffCC[NUM_COUNTERS];

  numBitmaps = BITS_PER_BASE * NUM_BITMAPS * K_STEPS;
  numCounters = NUM_COUNTERS;

  printf("NUM ENTRIES: real: %u saved: %u, size: %u\n", newfmi->bwtsize/fmi->chunk, newfmi->nentries, (uint32_t) sizeof(bitcnt_t));

  for (k = 0; k < NUM_COUNTERS; k++) {
    cntCC[k] = 0;
    diffCC[k] = 0;
  }

  for (i = 0; i < newfmi->nentries; i++){
    for (k = 0; k < NUM_COUNTERS; k++) {
      if (newfmi->h_index[i].cnt[k] > newfmi->bwtsize) cntCC[k]++;
      if (newfmi->h_index[i].cnt[k] !=  fmi->h_index[i].cnt[k]) {
        printf(" --------------------------------------------\n");
        printf("newEntry: %u, oldEntry: %u\n", newfmi->h_index[i].cnt[k], fmi->h_index[i].cnt[k]);
        printf("idxEntry: %u, idxCounter: %u \n", i, k);
        diffCC[k]++;
      }
    }
  }

  printf("Vector cntCC: \n");
  for (k = 0; k < NUM_COUNTERS; k++) {
    printf ("[%d]: %u \n", k, cntCC[k]);
  }
  printf("\n");

  printf("Vector diffCC: \n");
  for (k = 0; k < NUM_COUNTERS; k++) {
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
    for (i = 0; i < numCounters; i++){
      printf("B[%d]: %u - ", i, fmi->h_index[indexEntry+idx].cnt[i]);
    }
    printf("\n");
    printf("\n");

    printf("INDEX[%d] (NEW LAYOUT)\n", indexEntry+idx);
    printf("BITMAPS: ");
    for (i = 0; i < numBitmaps; i++){
      printf("B[%d]: %d - ", i, newfmi->h_index[indexEntry+idx].bitmap[i]);
    }
    printf("\n");

    printf("COUNTERS: ");
    for (i = 0; i < numCounters; i++){
      printf("C[%d]: %d - ", i, newfmi->h_index[indexEntry+idx].cnt[i]);
    }
    printf("\n");
  }
}

void transformIndex(void *index, void *newIndex)
{
  fmi_t *fmi    = (fmi_t *) index;
  fmi_t *newfmi = (fmi_t *) newIndex;
  uint32_t i, k, s;

  for (i = 0; i < newfmi->nentries; i++){
    for (k = 0; k < NUM_BITMAPS; k++) {
       for (s = 0; s < newfmi->steps; s++){
         newfmi->h_index[i].bitmap[(2 * newfmi->steps * k) + (2 * s)    ] = fmi->h_index[i].bitmap[(2 * s * NUM_BITMAPS) + k];
         newfmi->h_index[i].bitmap[(2 * newfmi->steps * k) + (2 * s) + 1] = fmi->h_index[i].bitmap[(2 * s * NUM_BITMAPS) + NUM_BITMAPS + k];
      }
    }
  }

  /* OLD INTERLEAVING
  for (i = 0; i < newfmi->nentries; i++){
    for (k = 0; k < NUM_BITMAPS; k++) {
      newfmi->h_index[i].bitmap[2*k]    = fmi->h_index[i].bitmap[k];
      newfmi->h_index[i].bitmap[2*k+1]  = fmi->h_index[i].bitmap[NUM_BITMAPS + k];
    }
  }*/

  for (i = 0; i < newfmi->nentries; i++)
    for (k = 0; k < NUM_COUNTERS; k++)
      newfmi->h_index[i].cnt[k] = fmi->h_index[i].cnt[k];
}

int32_t main(int32_t argc, char *argv[])
{
  char      *indexFile = argv[1];
  char      *newIndexFile = argv[1];
  uint32_t  indexEntry, numEntries;

  void *index = NULL;
  void *newIndex = NULL;
  int32_t error;

  if(argc > 2){
    indexEntry = atoi(argv[2]);
    numEntries = atoi(argv[3]);
  }

  error=loadIndex(indexFile, &index);
  HOST_HANDLE_ERROR(error);

  error=initIndex(index, &newIndex);
  HOST_HANDLE_ERROR(error);

  transformIndex(index, newIndex);

  if(argc > 2)
    checksumPrint(index, newIndex, indexEntry, numEntries);

  error = saveIndex(indexFile, newIndex);
  HOST_HANDLE_ERROR(error);

  error=freeIndex(&index);
  HOST_HANDLE_ERROR(error);

  error=freeIndex(&newIndex);
  HOST_HANDLE_ERROR(error);

  return (SUCCESS);
}
