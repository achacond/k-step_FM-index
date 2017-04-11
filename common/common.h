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

#ifndef COMMON_H_
#define COMMON_H_

#include <inttypes.h>
#include <stdint.h>

#ifndef MIN
  #define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif /* MIN */

/* macro to detect and to notify errors */
#define HOST_HANDLE_ERROR(error) {{if (error) { fprintf(stderr, "%s\n", errorCommon(error)); exit(EXIT_FAILURE); }}}
#define CUDA_HANDLE_ERROR(error) (HandleError(error, __FILE__, __LINE__ ))

typedef enum
{
  SUCCESS,
  E_OPENING_INDEX_FILE,
  E_ALLOCATING_BWT,
  E_ALLOCATING_FMI,
  E_READING_BWT,
  E_READING_FMI,
  E_SAVING_INDEX_FILE,
  E_SAVING_BWT_FILE,
  E_BUILDING_BWT,
  E_BUILDING_FMI,
  E_OPENING_REFERENCE_FILE,
  E_ALLOCATING_REFERENCE,
  E_READING_MFASTA_FILE,
  E_READING_REFERENCE_FILE,
  E_OPENING_MFASTA_FILE,
  E_ALLOCATING_MFASTA,
  E_ALLOCATING_RESULTS,
  E_OPENING_RESULTS_FILE,
  E_READING_RESULTS_FILE,
  E_NOT_IMPLEMENTED,
  E_INDEX_VER_BASELINE      = 100,
  E_INDEX_VER_INTERLEAVE    = 101,
  E_INDEX_VER_BASELINE_AC   = 200,
  E_INDEX_VER_INTERLEAVE_AC = 201,
} error_t;

typedef struct {
  uint32_t num;
  uint32_t size;
  char *h_queries;
  char *d_queries;
} qrys_t;

typedef struct {
  uint32_t size;
  char *h_reference;
  char *d_reference;
} ref_t;

typedef struct {
  uint32_t num;
  uint32_t *h_results;
  uint32_t *d_results;
} res_t;

double sampleTime();
inline static char *GETENV(char *envstr);

uint32_t base2index(uint32_t base);
int32_t  loadRef(char *fn, uint32_t refsize, void **reference);
int32_t  loadQueries(char *fn, uint sizequery, uint numqueries, void **queries);
int32_t  saveRef(char *fn, void *reference);
int32_t  readQueries(char *fn, uint sizequery, uint numqueries, unsigned char **queries);
int32_t  writeResults(char *fn, uint *results, uint numqueries);
int32_t  loadResults(char *fn, void **results);
int32_t  freeQueries(void **queries);
int32_t  freeResults(void **results);
int32_t  saveResults(char *fn, void *results, void *index);
char*    errorCommon(error_t e);

#endif /* COMMON_H_ */
