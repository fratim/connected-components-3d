#!/bin/bash


# ren preparation script (make folder)
python preparation.py


# eecutre step one computations for every block
for bz in {0..3}
do
    for by in {0..3}
    do
        for bx in {0..3}
        do

            python stepOne.py $bz $by $bx

        done
    done
done
