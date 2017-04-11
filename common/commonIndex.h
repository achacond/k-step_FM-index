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

#ifndef COMMONINDEX_H_
#define COMMONINDEX_H_

#include <inttypes.h>

#ifndef MIN
  #define MIN(_a, _b) (((_a) < (_b)) ? (_a) : (_b))
#endif /* MIN */

/* macro to detect and to notify errors */
#define IFERROR(error) {{if (error) { fprintf(stderr, "%s\n", errorIndex(error)); exit(EXIT_FAILURE); }}}
#define COMERROR(error) {{if (error) { fprintf(stderr, "%s\n", errorCommon(error)); exit(EXIT_FAILURE); }}}
#define MEMERROR(error) {{if (error==NULL) { fprintf(stderr, "Error reserva de mem. para resultados\n"); exit(EXIT_FAILURE); }}}
#define HANDLE_ERROR(error) (HandleError(error, __FILE__, __LINE__ ))

/*- Datatypes for BWT*/
typedef uint8_t sauchar_t;
typedef int32_t saint_t;
typedef int32_t saidx_t;
typedef int64_t saidx64_t;

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
char     *errorCommon(int e);

#endif /* COMMON_H_ */
