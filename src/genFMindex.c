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
#include "../resources/divsufsort.h"
#include "../common/common.h"
#include "../common/interface.h"

#if defined(K_STEPS) || defined(NUM_BITMAPS) || defined(NUM_COUNTERS)
#else
  #define K_STEPS      1
  #define NUM_CHUNK    64
  #define NUM_COUNTERS 4
#endif

#define OUT_INDEX_TAG  100

typedef struct {
  uint32_t cnt[NUM_COUNTERS];
} counters_t;

typedef struct {
  uint32_t bitmap[(NUM_CHUNK/16)*K_STEPS];
  uint32_t cnt[NUM_COUNTERS];
} bitcnt_t;

typedef struct {
  uint32_t  steps;
  uint32_t  bwtsize;
  uint32_t  ncounters;
  uint32_t  nentries;
  uint32_t  chunk;
  uint32_t  nbitmaps;
  uint32_t  dollarPositionBWT[K_STEPS];
  uint32_t  dollarBaseBWT[K_STEPS];
  char*     BWT[K_STEPS];
  bitcnt_t* index;
} fmi_t;

int32_t mod(int32_t x, int32_t m) {
  return (x%m + m)%m;
}

static inline char *GETENV(char *envstr)
{
  char *env = getenv(envstr);
  if (!env) return "0";
  else return env;
}

uint32_t base2index(uint32_t base)
{
  uint32_t flg2, flg3, bit1, bit0, index;

  flg2 = base & 0x02;
  flg3 = flg2 ^ 0x02;
  bit1 = base & 0x04;
  bit0 = (bit1) ? flg3 : flg2;

  index = bit1 | bit0;
  index >>= 1;

  return(index);
}

uint32_t index2BaseBWT(char **BWT, uint32_t position, uint32_t steps)
{  
  uint32_t base, indexBase=0x0, i;
  char *indexBWT;

  for(i=0; i<steps; i++){
    indexBWT = BWT[i];
    base = (uint32_t) indexBWT[position];
    indexBase |= base2index(base) << (2 * i); 
  }

  return indexBase;
}

uint32_t dollar2BaseBWT(char **BWT, uint32_t position, uint32_t nsteps, uint32_t step)
{  
  uint32_t base, indexBase = 0x0, i;
  char     *indexBWT;
  uint32_t mask = 0xFFFFFFFF;

  for(i = 0; i < nsteps; i++){
    indexBWT = BWT[i];
    base = (uint32_t) indexBWT[position];
    indexBase |= base2index(base) << (2 * i); 
  }  
  indexBase &= (mask << (2 * step));
  return indexBase;
}

uint32_t checkPositionBWT(uint32_t *dollarPositionBWT, uint32_t position, uint32_t steps)
{
  uint32_t flag = 1, i;
  for(i=0; i<steps; i++)
    flag = (position!=dollarPositionBWT[i]) && flag;
  return flag;  
}

int32_t saveBWT(char *fn, char *bwt, uint32_t refsize, uint32_t dollarPosition, uint32_t dollarBase)
{
  FILE *fp = NULL;
  char intro = '\n';
  int64_t sizeline = 70;
  int64_t cnt = 0;
  int64_t nlines = refsize / sizeline;

  fp = fopen(fn, "a");
  if (fp==NULL) return (E_SAVING_BWT_FILE);

  fprintf(fp, "BWT Dollar Position: %u\n", dollarPosition);
  fprintf(fp, "BWT Dollar Base: %u\n", dollarBase);
  fprintf(fp, "> %u", refsize);

  while(cnt<nlines){
    fwrite(&intro, sizeof(char), 1, fp);
    fwrite(bwt + (sizeline * cnt), sizeof(char), sizeline, fp);
    cnt++;
  }

  if((refsize%sizeline)!=0){ 
    fwrite(&intro, sizeof(char), 1, fp);
    fwrite(bwt + (sizeline * cnt), sizeof(char), refsize%sizeline, fp);
  }

  fwrite(&intro, sizeof(char), 1, fp);
  fclose(fp);
  return(SUCCESS);
}


int32_t saveIndex(const char *fn, void *index)
{
  fmi_t *fmi = (fmi_t *) index;
  uint32_t i, error, id_index = OUT_INDEX_TAG;
  char fmiFileOut[512];
  FILE *fp = NULL;

  sprintf(fmiFileOut, "%s.%u.%ufmi%usteps.fmi", fn, (fmi->bwtsize)-1, fmi->chunk, fmi->steps);

  fp = fopen(fmiFileOut, "wb");
  if (fp == NULL) return (E_SAVING_INDEX_FILE);

  fwrite(&id_index, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->steps, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->bwtsize, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->ncounters, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->nentries, sizeof(uint32_t), 1, fp);
  fwrite(&fmi->chunk, sizeof(uint32_t), 1, fp);
  for(i=0; i<fmi->steps; i++)
  fwrite(&fmi->dollarPositionBWT[i], sizeof(uint32_t), 1, fp);
  for(i=0; i<fmi->steps; i++)
  fwrite(&fmi->dollarBaseBWT[i], sizeof(uint32_t), 1, fp);

  fwrite(fmi->index, sizeof(bitcnt_t), fmi->nentries, fp);
  fclose(fp);
  return (SUCCESS);
}


int32_t precalculateBasesKSteps(char **BWT, uint32_t *dollarPositionBWT, uint32_t chunk, uint32_t bwtsize, uint32_t ncounters, uint32_t nbitmaps, uint32_t steps, bitcnt_t **FMindex)
{
  uint32_t count[ncounters], acc[ncounters];
  bitcnt_t *index = NULL;
  uint32_t nentries = (bwtsize % chunk) ? (bwtsize / chunk) + 1 : bwtsize / chunk;
  uint32_t divcounters = bwtsize / chunk, modcounters = bwtsize % chunk;
  uint32_t sum, dollarBase, dollarBase1, dollarBase0, indexBase, position, cnt, i, j;

  index = (bitcnt_t*) malloc(nentries * sizeof(bitcnt_t));
  if (index == NULL) return (E_ALLOCATING_FMI);

  // Initialize all FMI counters to 0s 
  for(j = 0; j < nentries; j++){
    for(i = 0; i < nbitmaps; i++){
      index[j].bitmap[i] = 0x0;
    }
    for(i=0; i<ncounters; i++){
      index[j].cnt[i] = 0;
    }
  }
  
  for(i = 0; i < ncounters; i++){
    count[i] = 0;
    acc[i] = 0;
  }

  for(j = 0; j < divcounters; j++){
  // Updates the counter for each interval
  for(i = 0; i < ncounters; i++)
    index[j].cnt[i]=count[i];
  // Counting each interval
  for(i = 0; i < chunk; i++){
    position = j * chunk + i;
    indexBase=index2BaseBWT(BWT, position, steps);
    if(checkPositionBWT(dollarPositionBWT, position, steps))
       count[indexBase]++;
    }
  }

  // Counting the last characters
  if(modcounters != 0) {
    // Updates the counter for each interval
    for(i = 0; i < ncounters; i++)
      index[j].cnt[i] = count[i];
    // Counting each interval
    for(i = 0; i < modcounters; i++){
      position=(divcounters * chunk) + i;
      indexBase=index2BaseBWT(BWT, position, steps);
      if(checkPositionBWT(dollarPositionBWT, position, steps))
        count[indexBase]++;
    }
   }  

  // Calculate the Cb counters -> Accumulating the both counters Cb and Occ
  for(j = 1; j < ncounters; j++){
    sum = 0;
    for(i = 0; i < j; i++)
      sum += count[i];
    acc[j] = sum;
  }
  
  // Adapting Cb counters to $ symbols from BWT
  for(i = 0; i < steps; i++){
    indexBase=dollar2BaseBWT(BWT, dollarPositionBWT[i], steps, i);
    for(j = indexBase; j < ncounters; j++)
      acc[j]++;
  }

  // Adding the Cb with OCC partial values for each counter
  for (j = 0; j < nentries; j++){
    for (i = 0; i < ncounters; i++)
      index[j].cnt[i] += acc[i];
  }

  (*FMindex) = index;
  return (SUCCESS);
}

int32_t precalculateBasesPreviousBWT(char *bwt, uint32_t chunk, uint32_t bwtsize, counters_t **FMcounters)
{
  uint32_t countA = 0, countC = 0, countG = 0, countT = 0;
  uint32_t accA = 0, accC = 0, accG = 0, accT = 0;
  counters_t *counters = NULL;
  uint32_t nentries = (bwtsize % chunk) ? (bwtsize / chunk) + 1 : bwtsize / chunk;
  uint32_t divcounters = bwtsize / chunk, modcounters = bwtsize % chunk;
  uint32_t cnt, i, j;
  char base;
    
  counters = (counters_t*) malloc(nentries * sizeof(counters_t));
  if (index == NULL) return (E_ALLOCATING_FMI);

  // Initialize all counters to 0s
  for(i = 0; i < nentries; i++){
    counters[i].cnt[0] = 0;
    counters[i].cnt[1] = 0;
    counters[i].cnt[2] = 0;
    counters[i].cnt[3] = 0;
  }

  for(j = 0; j < divcounters; j++){
    counters[j].cnt[0] = countA;
    counters[j].cnt[1] = countC;
    counters[j].cnt[2] = countG;
    counters[j].cnt[3] = countT;
    for(i = 0; i < chunk; i++){
      base=bwt[(j * chunk) + i];
      if(base == 'A') countA++;
      if(base == 'C') countC++;
      if(base == 'G') countG++;
      if(base == 'T') countT++;
    }
  }

  if(modcounters != 0){
    counters[j].cnt[0] = countA;
    counters[j].cnt[1] = countC;
    counters[j].cnt[2] = countG;
    counters[j].cnt[3] = countT;
    for(i = 0; i < modcounters; i++){
      base = bwt[(divcounters * chunk) + i];
      if(base=='A') countA++;
      if(base=='C') countC++;
      if(base=='G') countG++;
      if(base=='T') countT++;
    }
  }  
    
  accA = 1;
  accC = 1 + countA;
  accG = 1 + countA + countC;
  accT = 1 + countA + countC + countG;
    
  for (i = 0; i < nentries; i++){
    counters[i].cnt[0] += accA;
    counters[i].cnt[1] += accC;
    counters[i].cnt[2] += accG;
    counters[i].cnt[3] += accT;
  }
    
  (*FMcounters) = counters;
  return (SUCCESS);
}

int32_t generateOthersBWTs(char *ref, char *BWT0, counters_t *counters, int32_t steps, uint32_t chunk, uint32_t bwtsize, uint32_t dollarPositionBWT0, 
                           uint32_t *originaldollarPositionBWT, char **originalBWT) 
{  
  char base;
  char *BWT[steps];
  char *actualBWT=NULL;
  int64_t dollarPositionAllBWT[steps];
  int64_t cnt, despPosition, posBWT, i, j;
  int64_t newPosition, finalPosition, position = dollarPositionBWT0, refPosition;
  
  BWT[0] = BWT0;
  dollarPositionAllBWT[0] = dollarPositionBWT0;

  // Allocating memory for the the others k-BWTs (k-1 depths)
  for (i = 1; i < steps; i++){
      BWT[i] = (char*) malloc(bwtsize * sizeof(char));
      if (BWT[i] == NULL) return (E_BUILDING_FMI);
  }

  //REALIZA LF-MAPPING POR CADA CARACTER DEL BWT0
  for(refPosition = bwtsize - 1; refPosition >= 0; refPosition--){
  //RELLENAR NUEVOS BWTs: steps=4 -> refposition a 3 
    if(refPosition >= (steps - 1)){
      for (i = 1; i < steps; i++){
        actualBWT = BWT[i];
        actualBWT[position] = ref[refPosition - i];
      }
    }else{
    // The step k-1 is an special case, due to the $ symbol ends the reference
      for (i = 1; i < steps; i++){
        actualBWT = BWT[i];
        if(mod((refPosition-i), bwtsize) == (bwtsize - 1)) actualBWT[position] = '$';
          else actualBWT[position] = ref[mod((refPosition - i), bwtsize)];
      }
      dollarPositionAllBWT[refPosition+1]=position;
    }
    
    // Applying LF-MAPPING
    despPosition = position%chunk;
    posBWT = position - despPosition;

    base = BWT0[position];
    switch (base){
      case 'A':
        position = counters[position / chunk].cnt[0];
        break;
      case 'C':
        position = counters[position / chunk].cnt[1];
        break;
      case 'G':
        position = counters[position / chunk].cnt[2];
        break;
      case 'T':
        position = counters[position / chunk].cnt[3];
        break;
      default:
        position = 0;
      break;
    }
    // Counting locally the BWT chunk
    for(cnt = 0; cnt < despPosition; cnt++){
      if(BWT0[posBWT + cnt] == base)
        position++;
    }
  }

  // Saving results: BWTs & dollar positions
  for (i = 0; i < steps; i++){
    originaldollarPositionBWT[i] = (uint32_t) dollarPositionAllBWT[i];
    originalBWT[i] = BWT[i];
  }

  return (SUCCESS);
}

int32_t substring2bitmap(char *bwt, uint32_t numBases, uint32_t *bitmap1, uint32_t *bitmap0)
{
  uint32_t base, bit0, bit1, flg2, flg3, auxBitmap0 = 0x0, auxBitmap1 = 0x0;
  uint32_t nbit, i = 2, j = 1, sizeBitmap = 32;

  for(nbit = 0; nbit < numBases; nbit++) {
    base = (uint32_t) bwt[nbit];
    flg2 = base & 0x02;
    flg3 = flg2 ^ 0x02;
    bit1 = base & 0x04;
    bit0 = (bit1) ? flg3 : flg2;

    bit1=(bit1 >> i) << (sizeBitmap - nbit - 1);
    bit0=(bit0 >> j) << (sizeBitmap - nbit - 1);

    auxBitmap1 |= bit1;
    auxBitmap0 |= bit0;
  }

  (*bitmap1) = auxBitmap1;
  (*bitmap0) = auxBitmap0;
  return 0;
}

//It generates consecutive bitmaps -> |#|BWT0-Bitmap0|BWT0-Bitmap1|...|BWT0-BitmapN|#|...|#|BWTN-Bitmap0|BWTN-Bitmap1|...|BWTN-BitmapN|#|
int32_t bwt2bin(char **BWT, uint32_t chunk, uint32_t size, uint32_t steps, uint32_t nbitmaps, bitcnt_t *index)
{
  uint32_t numBases = 32, bitsxletter = 2, ncolumn = nbitmaps / bitsxletter, modBitmaps = size % numBases, restBitmaps = size - modBitmaps;
  uint64_t pos = 0, i = 0, j = 0, k = 0;
  char *indexBWT;

  while(pos < restBitmaps) {
    for(k = 0; k < steps; k++){
      indexBWT = BWT[k];
      substring2bitmap(&indexBWT[pos], numBases, &index[i].bitmap[(k * nbitmaps) + ncolumn + j], &index[i].bitmap[(k * nbitmaps) + j]);
    }
    j++; 
    pos += numBases; // Advance 32 bases (bitmap to bitmap)
    if((pos % chunk) == 0) {
      i++;
      j = 0;
    }
  }

  if(modBitmaps){
    // Last iteration (the ending: 32 bases + padding) 
    for(k=0; k<steps; k++){
      indexBWT=BWT[k];
      substring2bitmap(&indexBWT[pos], modBitmaps, &index[i].bitmap[(k*nbitmaps)+(1*ncolumn)+j], &index[i].bitmap[(k*nbitmaps)+(0*ncolumn)+j]);
    }
  }

  return (SUCCESS);
}

int32_t buildIndex(void *reference, void **index)
{
  fmi_t *fmi = (fmi_t*) malloc(sizeof(fmi_t));
  ref_t *ref = (ref_t *) reference;
  char bwtFileOut[512];
  FILE *fp = NULL;
  char *actualBWT = NULL;

  saidx64_t *tmpsa = NULL;
  sauchar_t *tmpbwt = NULL;
  counters_t *auxCounters = NULL;
  uint32_t tmpbwtsize = ref->size, chunkGen = 32, i, j;
  uint32_t dollarPosition;
  int32_t  error;

  fmi->steps = K_STEPS;
  fmi->bwtsize = ref->size + 1;
  fmi->chunk = NUM_CHUNK;
  fmi->nbitmaps = (NUM_CHUNK / 32) * 2;
  fmi->ncounters = NUM_COUNTERS;
  fmi->nentries = (fmi->bwtsize % fmi->chunk) ? (fmi->bwtsize / fmi->chunk) + 1 : fmi->bwtsize / fmi->chunk;

  // Building the first BWT (BWT(0))
  tmpbwt = (char*) malloc(tmpbwtsize * sizeof(char));
  if(tmpbwt == NULL) return (E_ALLOCATING_BWT);
  dollarPosition = (uint32_t) divbwt64(ref->h_reference, tmpbwt, tmpsa, tmpbwtsize);
  if(dollarPosition < 0) return (E_BUILDING_BWT);
  fmi->BWT[0] = (char*) malloc((fmi->bwtsize) * sizeof(char));
  if(fmi->BWT[0] == NULL) return (E_BUILDING_FMI);

  // Adding the $ symbol to BWT(0)
  memcpy(fmi->BWT[0], tmpbwt, dollarPosition);
  (fmi->BWT[0])[dollarPosition] = '$';
  memcpy((fmi->BWT[0]) + dollarPosition + 1, (tmpbwt + dollarPosition), tmpbwtsize - dollarPosition);
  free(tmpbwt);

  // Saving $ position from BWT0
  fmi->dollarPositionBWT[0] = dollarPosition;

  // Generating the rest depths of BWT
  if(fmi->steps > 1){
    error=precalculateBasesPreviousBWT(fmi->BWT[0], chunkGen, fmi->bwtsize, &auxCounters);
    HOST_HANDLE_ERROR(error);
    error=generateOthersBWTs(ref->h_reference, fmi->BWT[0], auxCounters, fmi->steps, chunkGen, fmi->bwtsize, fmi->dollarPositionBWT[0], fmi->dollarPositionBWT, fmi->BWT);
    HOST_HANDLE_ERROR(error);
    free(auxCounters);
  }
  
  // Tricky stuff: $ are represented by A symbols inside BMP
  for(i = 0; i < fmi->steps; i++){
    actualBWT=fmi->BWT[i];
    actualBWT[fmi->dollarPositionBWT[i]] = 'A';
  } 

  // Calculate counters (rLF) and bitmaps (BMP)
  error = precalculateBasesKSteps(fmi->BWT, fmi->dollarPositionBWT, fmi->chunk, fmi->bwtsize, fmi->ncounters, fmi->nbitmaps*fmi->steps, fmi->steps, &fmi->index);
  HOST_HANDLE_ERROR(error);
  error = bwt2bin(fmi->BWT, fmi->chunk, fmi->bwtsize, fmi->steps, fmi->nbitmaps, fmi->index);
  HOST_HANDLE_ERROR(error);

  // Save $ positions from each k BWT
  for(i=0; i<fmi->steps; i++){
    fmi->dollarBaseBWT[i]=index2BaseBWT(fmi->BWT, fmi->dollarPositionBWT[i], fmi->steps);
  }

  // For debug purposes: save in a file each k BWT 
  for(i = 0; i < fmi->steps; i++){
    if(atoi(GETENV("INDEX_DGB"))){
      actualBWT = fmi->BWT[i];
      actualBWT[fmi->dollarPositionBWT[i]] = '$';
      printf("dollarBase - %u, dollarPosition - %u, bwt%u:\n", fmi->dollarBaseBWT[i], fmi->dollarPositionBWT[i], i);
      for(j = 0; j < fmi->bwtsize; j++)
        printf("%c", actualBWT[j]);
      printf("\n");
      sprintf(bwtFileOut, "%u.BWT%u", (fmi->bwtsize)-1, i);    
      error = saveBWT(bwtFileOut, fmi->BWT[i], fmi->bwtsize, fmi->dollarPositionBWT[i], fmi->dollarBaseBWT[i]);
      HOST_HANDLE_ERROR(error);
    }
  }

  // Freeing memory
  for(i = 0; i < fmi->steps; i++)
    free(fmi->BWT[i]);

  (*index) = fmi;
  return (SUCCESS);
}

int32_t freeIndex(void **index) 
{  
  fmi_t *fmi = (fmi_t *) (*index);
  if(fmi->index != NULL){
    free(fmi->index);
    fmi->index = NULL;
  }
  return(SUCCESS);
}

