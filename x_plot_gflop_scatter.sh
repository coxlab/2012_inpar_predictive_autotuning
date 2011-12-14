#!/bin/bash

# hacky but should get stuff done
for a in 480 580 1060 2070; do
    python fig.py gflop_scatter munctional0 $a 0.5 200;
done;

python fig.py gflop_scatter vader 295 0.5 200

