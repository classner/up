#!/usr/bin/zsh

trap "exit" INT

for rangestart in 0 $(seq 10 3 82); do
    echo Running training for ${rangestart}
    ./fit_forest.py $1 ${rangestart} --finalize
done
