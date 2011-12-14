#!/bin/bash

machine=munctional0
what=580
#machine=vader
#what=295

python fig.py allstars_mixup timings/${machine}/timing1_${what}_big.pkl timings/${machine}/timing1_${what}_big.pkl_allstars_mixup_timings

mv -vf fig_allstars_mixup_${what}.pdf paper
