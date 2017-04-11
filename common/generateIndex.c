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

#ifndef GENERATEINDEX_C
#define GENERATEINDEX_C

#include <stdio.h>
#include <stdlib.h>
#include "common.h"
#include "interface.h"

int32_t main(int32_t argc, char *argv[])
{
  void *reference;
  void *index;
  unsigned char *refFile = argv[1];
  uint32_t refsize = (uint) atoll(argv[2]);
  int32_t error;

  error = loadRef(refFile, refsize, &reference);
  HOST_HANDLE_ERROR(error);

  error = buildIndex(reference, &index);
  HOST_HANDLE_ERROR(error);

  error = saveIndex(refFile, index);
  HOST_HANDLE_ERROR(error);
  error = saveRef(refFile, reference);
  HOST_HANDLE_ERROR(error);
  
  error = freeIndex(&index);
  HOST_HANDLE_ERROR(error);
  error = freeReference(&reference, &index);
  HOST_HANDLE_ERROR(error);

  return (SUCCESS);
}

#endif /* GENERATEINDEX_C */
