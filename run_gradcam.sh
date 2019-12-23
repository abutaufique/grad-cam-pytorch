#!bin/bash
txtfile=/home/at7133/Research/Domain_adaptation/OPDA_BP/da_success.txt
python main.py demo3 \
    -i $txtfile \
    -o opda_wda \
    -n 6 \
    -m OPDA
