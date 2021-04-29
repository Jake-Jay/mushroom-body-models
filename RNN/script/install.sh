#!/bin/bash -i
export JULIA_PROJECT=$(realpath $(dirname ${BASH_SOURCE[0]}))
export PATH=$(echo "$PATH" | tr ":" "\n" | grep -v "$JULIA_PROJECT" | tr "\n" ":")
julia -e 'using Pkg; Pkg.instantiate(); Pkg.build()'
if ! grep PROMPT_COMMAND ~/.bashrc &> /dev/null;  then
    cat >> ~/.bashrc << "EOF"
PROMPT_COMMAND='if [[ "$bashrc" != "$PWD" && "$PWD" != "$HOME" && -e .bashrc ]]; then bashrc="$PWD"; . .bashrc; fi'
EOF
fi
julia --startup-file=no $JULIA_PROJECT/src/compile.jl