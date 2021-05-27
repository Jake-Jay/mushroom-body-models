export JULIA_PROJECT=$(realpath $(dirname ${BASH_SOURCE[0]}))
export PATH=$JULIA_PROJECT:$PATH
export PATH=$JULIA_PROJECT/script:$PATH
export GKSwstype=100