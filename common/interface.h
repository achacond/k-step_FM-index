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

#ifndef INTERFACE_H_
#define INTERFACE_H_

#include <stdint.h>

int32_t loadIndex(const char *fn, void **index);
int32_t saveIndex(const char *fn, void *index);
int32_t initResults(uint32_t numresults, void **results);
void    searchIndexCPU(void *index, void *queries, void *resIntervals);
void    searchIndexGPU(void *index, void *queries, void *resIntervals);
void    searchIndex(void *index, uint32_t numqueries, char *queries, uint32_t qrysize, uint32_t *resIntervals);
int32_t freeIndex(void **index);
int32_t freeReference(void **reference, void **index);
int32_t buildIndex(void *reference, void **index);
int32_t freeQueriesGPU(void **queries);
int32_t freeResultsGPU(void **results);
int32_t freeIndexGPU(void **index);
int32_t transferGPUtoCPU(void *results);
int32_t transferCPUtoGPU(void *index, void *queries, void *results);
char*   errorIndex(int32_t e);

#endif /* INTERFACE_H_ */
