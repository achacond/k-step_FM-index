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
#include <stdint.h>
#include "interface.h"
#include "common.h"
#include "omp.h"
#ifdef PROFILE
  #include <likwid.h>
#endif


int32_t main(int32_t argc, char *argv[])
{
  char *indexFile = argv[1];
  char *qryFile   = argv[2];
  char *resFile   = argv[1];

  uint qrysize    = (uint) atoll(argv[3]);
  uint numqueries = (uint) atoll(argv[4]);

  void *index   = NULL;
  void *queries = NULL;
  void *results = NULL;

  double   ts,ts1,total;
  uint32_t iter = 5, n;
  int32_t  error;

  error = loadIndex(indexFile, &index);
  HOST_HANDLE_ERROR(error);

  error = loadQueries(qryFile, qrysize, numqueries, &queries);
  HOST_HANDLE_ERROR(error);

  error = initResults(numqueries, &results);
  HOST_HANDLE_ERROR(error);

  #ifdef CUDA
    error = transferCPUtoGPU(index, queries, results);
    HOST_HANDLE_ERROR(error);
  #endif

  #ifdef PROFILE
    likwid_markerInit();
  #endif
      
  ts = sampleTime();

    #ifdef CUDA
      for(n = 0; n < iter; n++)
        searchIndexGPU(index, queries, results);
    #else
      #pragma omp parallel private(n)
      {
        #ifdef PROFILE
  			  likwid_markerStartRegion("Search Queries");
        #endif

        for(n = 0; n < iter; n++)
          searchIndexCPU(index, queries, results);

	    #ifdef PROFILE
          likwid_markerStopRegion("Search Queries");
        #endif      
      }
    #endif

  ts1 = sampleTime();

  #ifdef PROFILE
	  likwid_markerClose();
  #endif      

  #ifdef CUDA
    error = transferGPUtoCPU(results);
    HOST_HANDLE_ERROR(error);
  #endif

  error = saveResults(resFile, results, index);
  HOST_HANDLE_ERROR(error);

  #ifdef CUDA
    error = freeIndexGPU(&index);
    HOST_HANDLE_ERROR(error);
    error = freeQueriesGPU(&queries);
    HOST_HANDLE_ERROR(error);
    error = freeResultsGPU(&results);
    HOST_HANDLE_ERROR(error);
  #endif

  total = (ts1 - ts) / iter;
  printf("TIME: \t %f \n", total);

  error = freeIndex(&index);
  HOST_HANDLE_ERROR(error);
  error = freeQueries(&queries);
  HOST_HANDLE_ERROR(error);
  error = freeResults(&results);
  HOST_HANDLE_ERROR(error);

  return (SUCCESS);
}
