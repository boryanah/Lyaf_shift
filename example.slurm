#!/bin/bash
##SBATCH --qos=debug
#SBATCH --qos=regular
##SBATCH --time=12:00:00 # doesn't seem enough...
#SBATCH --time=05:00:00
##SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=cpu
##SBATCH --account=m1727

. "/global/homes/b/boryanah/anaconda3/etc/profile.d/conda.sh"
conda activate desc

#source /global/common/software/desi/desi_environment.sh 23.1 # inherits it
#source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
#module unload desiutil
#module load desiutil/3.2.6
#alias emacs="env --unset=LD_LIBRARY_PATH emacs"
#alias more="less"

sh tmp_run_corr_fft.sh
