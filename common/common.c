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

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"

double sampleTime()
{
  struct timespec tv;
  clock_gettime(CLOCK_REALTIME, &tv);
  return((tv.tv_sec+tv.tv_nsec/1000000000.0));
}

char *GETENV(char *envstr)
{
  char *env = getenv(envstr);
  if (!env) return "0";
  else return env;
}

int readRef(char *fn, uint32_t refsize, char **reference)
{
  FILE *fp = NULL;
  char cadena[256];
  int64_t countdown = refsize;
  int64_t cleidos = 0;
  int64_t cguardados = 0;
  int64_t pos = 0;
  char *ref = NULL;
  char *res = NULL;

  fp = fopen(fn, "rb");
  if (fp==NULL) return (E_OPENING_REFERENCE_FILE);

  ref = (char*) malloc(refsize * sizeof(char));
  if (ref==NULL) return (E_ALLOCATING_REFERENCE);

  res = fgets(cadena, 256, fp);
  if(res == NULL) return(E_READING_REFERENCE_FILE);

  if (cadena[0] != '>') return (E_READING_MFASTA_FILE);

  while((fgets(cadena, 256, fp) != NULL)&&(countdown>0)){
    cleidos = strlen(cadena);
    if(cleidos) cleidos--;
    cguardados = MIN(cleidos, countdown);
    memcpy((ref + pos), cadena, cguardados);
    pos =  pos + cguardados;
    countdown = countdown - cguardados;
  }

  fclose(fp);
  (*reference) = ref;
  return (SUCCESS);
}

int32_t loadRef(char *fn, uint32_t refsize, void **reference)
{
  ref_t *ref = (ref_t *) malloc(sizeof(ref_t));
  ref->size = refsize;
  readRef(fn, refsize, &ref->h_reference);
  ref->d_reference = NULL;
  (*reference) = ref;
  return (SUCCESS);
}

int32_t writeRef(char *fn, uint32_t refsize, char *ref)
{
  FILE *fp = NULL;
  char intro = '\n';
  int64_t sizeline = 70;
  int64_t cnt = 0;
  int64_t nlines = refsize / sizeline;
  char head[256];

  fp = fopen(fn, "wb");
  if (fp==NULL) return (E_OPENING_REFERENCE_FILE);

  sprintf(head, "> %u", refsize);
  fputs(head, fp);

  while(cnt<nlines){
    fwrite(&intro, sizeof(char), 1, fp);
    fwrite(ref+(sizeline*cnt), sizeof(char), sizeline, fp);
    cnt++;
  }

  if((refsize%sizeline)!=0){
    fwrite(&intro, sizeof(char), 1, fp);
    fwrite(ref + (sizeline * cnt), sizeof(char), refsize % sizeline, fp);
  }

  fwrite(&intro, sizeof(char), 1, fp);
  fclose(fp);
  return (SUCCESS);
}

int32_t saveRef(char *fn, void *reference)
{
  unsigned char refFileOut[512];
  int32_t error;
  ref_t *ref = (ref_t *) reference;

  sprintf(refFileOut, "%s.%u.fa", fn, ref->size);
  error = writeRef(refFileOut, ref->size, ref->h_reference);
  HOST_HANDLE_ERROR(error);

  return (SUCCESS);
}

int32_t loadQueries(char *fn, uint sizeQuery, uint32_t numQueries, void **queries)
{
  qrys_t *qrys = (qrys_t*)malloc(sizeof(qrys_t));

  FILE *fp = NULL;
  char cadena[1024];
  uint32_t cleidos = 0;
  uint32_t pos = 0;
  uint32_t i = 0;

  #ifdef INTERLEAVING_QUERIES
    //Architecture Dependent (128 Bytes / 32 ThreadsxWarp = 4 Bytes)
    //The number of queries have be multiple of sizeWarp.
    uint32_t idChar = 0;
    uint32_t sizeQueryWord = 4;
    uint32_t idQuery = 0;
    int32_t  idWord = 0;
    uint32_t idWarp = 0;
    uint32_t sizeWarp = 32;
    uint32_t index = 0;
    uint32_t posQueryChunk = 0;
    uint32_t numWarps = (numQueries/sizeWarp)+((numQueries%sizeWarp)?1:0);
    char*    deviceQueries = NULL;
    int32_t j;
  #endif

  fp = fopen(fn, "rb");
  if (fp == NULL) return (E_READING_REFERENCE_FILE);

  qrys->num = numQueries;
  qrys->size = sizeQuery;
  qrys->h_queries = (char*) malloc(sizeQuery * numQueries * sizeof(char));
  if ((qrys->h_queries) == NULL) return (E_ALLOCATING_MFASTA);
  qrys->d_queries = NULL;

  while (fgets(cadena, 1024, fp) != NULL){
    if (cadena[0] == '>') continue;
    cleidos = strlen(cadena);
    memcpy((qrys->h_queries)+pos, cadena, cleidos-1);
    pos = pos+cleidos-1;
    i++;
  }

  #ifdef INTERLEAVING_QUERIES
    //You have to review this malloc if yours queries have different size
    deviceQueries = (char*) malloc(sizeQuery * numQueries * sizeof(char));
      if (deviceQueries == NULL) return (E_ALLOCATING_MFASTA);
    //This code not run properly if sizequery%4 != 0
    for(idWarp = 0; idWarp < numWarps; idWarp++){
      posQueryChunk = idWarp * sizeWarp * sizeQuery;
      index = 0;
      for(idWord = sizeQuery - sizeQueryWord; idWord >= 0; idWord -= sizeQueryWord){
        for(idQuery = 0; idQuery < sizeWarp*sizeQuery; idQuery += sizeQuery){
          for(idChar = 0; idChar < sizeQueryWord; idChar++){
            deviceQueries[posQueryChunk + index] = qrys->h_queries[posQueryChunk + idQuery + idWord + idChar];
            index++;
          }
        }
      }
    }
    free(qrys->h_queries);
    qrys->h_queries = deviceQueries;
  #endif

  fclose(fp);
  (*queries) = qrys;
  return (SUCCESS);
}

int32_t writeResults(char *fn, uint32_t *results, uint32_t numqueries)
{
  FILE *fp=NULL;
  char cadena[256];
  uint32_t i;

  fp = fopen(fn, "w");
  if (fp == NULL) return (47);

  sprintf(cadena, "%u\n", numqueries);
  fputs(cadena, fp);

  for (i=0; i<numqueries; i++){
    sprintf(cadena, "%u %u\n", results[2*i], results[(2*i)+1]);
    fputs(cadena, fp);
  }

  fclose(fp);
  return (SUCCESS);
}

int32_t loadResults(char *fn, void **results)
{
  res_t *res = (res_t*) malloc(sizeof(res_t));
  FILE *fp;
  uint32_t i;
  int32_t fres = 0;

  fp = fopen(fn, "r");
  if (fp == NULL) return (E_OPENING_RESULTS_FILE);

  fres = fscanf(fp, "%u\n", &res->num);
  if(fres != 1) return(E_READING_RESULTS_FILE);

  res->h_results = (uint32_t*) malloc(2 * (res->num) * sizeof(uint32_t));
  res->d_results = NULL;

  for (i = 0; i < res->num; i++){
    fres = fscanf(fp, "%u %u\n", &res->h_results[2*i], &res->h_results[(2*i)+1]);
    if(fres != 2) return(E_READING_RESULTS_FILE);
  }

  fclose(fp);
  (*results) = res;
  return (SUCCESS);
}

int32_t initResults(uint numresults, void **results)
{
  res_t *res = (res_t*) malloc(sizeof(res_t));

  res->num = numresults;
  res->h_results = (uint32_t*) malloc(2 * numresults * sizeof(uint32_t));
   memset(res->h_results, 0, 2 * numresults * sizeof(uint32_t));
  if (res->h_results == NULL) return (E_ALLOCATING_RESULTS);
  res->d_results = NULL;

  (*results) = res;
  return (SUCCESS);
}

int32_t freeQueries(void **queries)
{
  qrys_t *qrys = (qrys_t *) (*queries);
  if(qrys->h_queries != NULL){
    free(qrys->h_queries);
    qrys->h_queries = NULL;
  }
  return(SUCCESS);
}

int32_t freeResults(void **results)
{
  res_t *res = (res_t *) (*results);
  if(res->h_results != NULL){
    free(res->h_results);
    res->h_results = NULL;
  }
  return(SUCCESS);
}

char *errorCommon(error_t e){
  switch(e) {
    case SUCCESS:   			            return "No error"; break;
    case E_OPENING_INDEX_FILE:        return "Cannot open index file"; break;               // 1
    case E_ALLOCATING_BWT:            return "Cannot allocate memory for bwt"; break;       // 2 21
    case E_ALLOCATING_FMI:            return "Cannot allocate memory for counters"; break;  // 3 20
    case E_READING_BWT:               return "Error reading index bwt"; break; //4
    case E_READING_FMI:               return "Error reading index counters"; break; //5
    case E_SAVING_INDEX_FILE:         return "Cannot open index file for save"; break; //8
    case E_SAVING_BWT_FILE:           return "Cannot open bwt file for save"; break; //9 y 15
    case E_BUILDING_BWT:              return "Error building bwt"; break; // 22
    case E_BUILDING_FMI:              return "Error building FMI, cannot allocate memory for bwt"; break; // 23
    case E_OPENING_REFERENCE_FILE:    return "Cannot open reference file"; break; // 30 37
    case E_ALLOCATING_REFERENCE:      return "Cannot allocate reference"; break; //31
    case E_READING_MFASTA_FILE:       return "Reference file isn't MFASTA format"; break; //32
    case E_READING_REFERENCE_FILE:    return "Error reading reference file"; break; //33
    case E_OPENING_MFASTA_FILE:       return "Cannot open MFASTS queries file"; break; //42
    case E_ALLOCATING_MFASTA:         return "Cannot allocate MFASTA queries"; break; //43
    case E_ALLOCATING_RESULTS:        return "Cannot allocate results"; break; //45
    case E_OPENING_RESULTS_FILE:      return "Cannot open results file for load intervals"; break; //48
    case E_READING_RESULTS_FILE:      return "Error reading results"; break; //49
    case E_NOT_IMPLEMENTED:           return "Not implemented"; break; //99
    case E_INDEX_VER_BASELINE: return "Error in the index type, use gfmiBaseLine_*Bases_*Step to generate an index_name.fmi type"; break; //100
    case E_INDEX_VER_INTERLEAVE: return "Error in the index type, use tfmiBMP_*Bases_*Step to generate an index_name.fmi.interleaving type"; break; //101
    case E_INDEX_VER_BASELINE_AC: return "Error in the index type, use tfmiAC_*Bases_*Step to generate an index_name.fmi.ac type"; break; //200
    case E_INDEX_VER_INTERLEAVE_AC: return "Error in the index type, use tfmiAC_*Bases_*Step to generate an index_name.fmi.interleaving.ac type"; break; //201
    default:  return "Unknown error";
  }
}


int32_t freeReference(void **reference, void **index)
{
  ref_t *ref = (ref_t *) (*reference);

  if(ref->h_reference != NULL)
    free(ref->h_reference);

  ref->h_reference=NULL;
  return (SUCCESS);
}

int32_t saveResults(char *fn, void *results, void *index)
{
  res_t *res = (res_t *) results;

  char resultsFileOut[512];
  int32_t error;

  #ifdef CUDA
    sprintf(resultsFileOut, "%s.res.gpu", fn);
  #else
    sprintf(resultsFileOut, "%s.res.cpu", fn);
  #endif

  error = writeResults(resultsFileOut, res->h_results, res->num);
  HOST_HANDLE_ERROR(error);

  return (SUCCESS);
}
