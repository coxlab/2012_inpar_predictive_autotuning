#!/bin/bash

python fig.py allstars_mixup timings/munctional0/timing1_580_big.pkl timings/munctional0/timing1_580_big.pkl_allstars_mixup_timings

mv -vf fig_allstars_mixup_580.pdf paper
