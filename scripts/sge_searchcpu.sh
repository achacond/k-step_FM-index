#!/bin/bash
##
##	k-step FM-index (benchmarking for CPU and GPU)
##	Copyright (c) 2011-2017 by Alejandro Chacon	<alejandro.chacond@gmail.com>
##
##	This program is free software: you can redistribute it and/or modify
##	it under the terms of the GNU General Public License as published by
##	the Free Software Foundation, either version 3 of the License, or
##	(at your option) any later version.
##
##	This program is distributed in the hope that it will be useful,
##	but WITHOUT ANY WARRANTY; without even the implied warranty of
##	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
##	GNU General Public License for more details.
##
##	You should have received a copy of the GNU General Public License
##	along with this program.	If not, see <http://www.gnu.org/licenses/>.
##
## PROJECT: k-step FM-index (benchmarking for CPU and GPU)
## AUTHOR(S): Alejandro Chacon <alejandro.chacond@gmail.com>
##

#$ -N CPU_SEARCH
#$ -S /bin/sh
#$ -cwd
#$ -q research.q
#$ -l excl=true
#$ -o cpu_search.profile.out
#$ -e cpu_search.profile.out



#REFERENCE CONFIGURATION
sizeref="500 2000 5000 20000 60000 200000 600000 2000000 8000000 50000000 100000000 400000000 750000000 1500000000"

#INDEX CONFIGURATION
reference_name="Human_Reference.fa"
bases="32"
ksteps="1 2 3 4"
indexExt="fmi.ac"
bin_extension="-ac-profile"

#QUERIES CONFIGURATION
numqueries="10000000"
sizequeries="120"

cd ..

module load cuda/7.5
module load gcc/4.9.0
module load likwid/4.0.1
(make clean all_cpu_searchers_profile install) > /dev/null 

echo "---------- START: CPU SEARCH INDEX PROFILE ----------"
for i in $sizeref
do
	for b in $bases
	do
		for s in $ksteps
		do
			binary_name=bin/fmIndexSearchCPU_"$b"bases_"$s"step"$bin_extension"
      index_name=data/"$reference_name"."$i"."$b"fmi"$s"steps."$indexExt"
      query_name=data/Q-"$numqueries"_B-"$sizequeries"_R-"$i".qry
			echo "$binary_name $index_name $query_name $sizequeries $numqueries"
	    $binary_name $index_name $query_name $sizequeries $numqueries
		done
	done
done
echo "---------- END: CPU SEARCH INDEX PROFILE ---------- "



