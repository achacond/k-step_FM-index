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

#$ -N Gen_Indexes
#$ -S /bin/sh
#$ -cwd
#$ -q research.q


#module load cuda/7.5
#module load gcc/4.9.0
#make clean install

#WORKLOAD CONFIGURATION
reference_name="data/references/Human_Reference.fa"
sizequeries="120"
#sizeref="500 2000 5000 20000 60000 200000 600000 2000000 8000000 50000000 100000000 400000000 750000000 1500000000 3000000000"
sizeref="750000000 3000000000"
numqueries="10000000"

cd ..

echo "---------- START BUILDING QUERIES ----------"
for idref in $sizeref
do
	echo "---------- Generating Queries (len = $idref)  ----------"
	bin/genreads.py $reference_name.$idref.fa $sizequeries $numqueries > data/Q-"$numqueries"_B-"$sizequeries"_R"$idref".qry
done
echo "---------- END BUILDING QUERIES ---------- "

cd scripts/

