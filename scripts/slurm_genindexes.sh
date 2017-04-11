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

#SBATCH --exclusive
#SBATCH --job-name="GEN_INDEXES"
#SBATCH -w batman
 
#SBATCH --time=10:00:00
#SBATCH --partition=p_hpca4se 
#SBATCH --mem=110g

#WORKLOAD CONFIGURATION
reference_name="data/Human_Reference.fa"
ksteps="1 2 3 4"

cd ../

module load cuda/7.5
module load gcc/4.9.1
make clean all_builders install

#sizeref="500 2000 5000 20000 60000 200000 600000 2000000 8000000 50000000 100000000 400000000 750000000 1500000000 3000000000"
sizeref="750000000 1500000000 3000000000"
bases="32"

echo "BUILDING STAGE 1"

echo "BUILDING Normal indexes for all sizes"
for stps in $ksteps
do
  for b in $bases
  do
	  for s in $sizeref
	  do
      echo "===> bin/gfmiBaseLine_"$b"bases_"$stps"step "$reference_name" "$s""
		  time bin/gfmiBaseLine_"$b"bases_"$stps"step "$reference_name" "$s"
	  done
  done
done

echo "BUILDING Aternate counting indexes for all sizes"
for stps in $ksteps
do
  for b in $bases
  do
	  for s in $sizeref
	  do
      echo "===> bin/tfmiAC_"$b"bases_"$stps"step "$reference_name"."$s"."$b"fmi"$stps"steps.fmi"
		  time bin/tfmiAC_"$b"bases_"$stps"step "$reference_name"."$s"."$b"fmi"$stps"steps.fmi
	  done
  done
done

echo "BUILDING STAGE 2"

sizeref="1500000000 3000000000"
bases="32 64 128 256"

echo "BUILDING Normal indexes for bases=$bases"
for stps in $ksteps
do
  for b in $bases
  do
	  for s in $sizeref
	  do
      echo "===> bin/gfmiBaseLine_"$b"bases_"$stps"step "$reference_name" "$s""
		  time bin/gfmiBaseLine_"$b"bases_"$stps"step "$reference_name" "$s"
    done
  done
done

echo "BUILDING Aternate counting indexes for bases=$bases"
for stps in $ksteps
do
  for b in $bases
  do
	  for s in $sizeref
	  do
      echo "===> bin/tfmiAC_"$b"bases_"$stps"step "$reference_name"."$s"."$b"fmi"$stps"steps.fmi"
		  time bin/tfmiAC_"$b"bases_"$stps"step "$reference_name"."$s"."$b"fmi"$stps"steps.fmi
	  done
  done
done

cd scripts/

