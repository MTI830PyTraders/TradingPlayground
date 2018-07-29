#!/bin/bash

declare -a arr=("pe_MCD"
		"pb_MCD"
		"de_MCD"
		"peg_MCD"
		"fcf_MCD"
		"pe_DIS"
		"pb_DIS"
		"de_DIS"
		"peg_DIS"
		"fcf_DIS"
		"pe_INTC"
		"pb_INTC"
		"de_INTC"
		"peg_INTC"
		"fcf_INTC"
		"pe_AAPL"
		"pb_AAPL"
		"de_AAPL"
		"peg_AAPL"
		"fcf_AAPL"
		"pe_MSFT"
		"pb_MSFT"
		"de_MSFT"
		"peg_MSFT"
		"fcf_MSFT"
		"pe_WDC"
		"pb_WDC"
		"de_WDC"
		"peg_WDC"
		"fcf_WDC"
	       )

for i in "${arr[@]}"
do
    export LSTM_PROFILE=$i; python3 main.py
    # or do whatever with individual element of the array
done
