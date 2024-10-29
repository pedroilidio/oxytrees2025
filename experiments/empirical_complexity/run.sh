#!/env/bin bash

# This script's parent directory
BASEDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python $BASEDIR/time_vs_n_artificial_data.py \
    --cache-dir $BASEDIR/cache \
    --outdir $BASEDIR/results \
    --n-jobs 4 \
    --random-state 0

