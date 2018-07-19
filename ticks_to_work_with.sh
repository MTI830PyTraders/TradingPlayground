#!/bin/bash

# NS1 cleanup
# Remove NS1 ticks without data
# Keep only US ticks
egrep '^(\w){1,8}_US'  NS1_20180715.csv | sed -r 's/^(\w{1,8})_US/\1/g' |  awk -F ',' '$6!="0.0" {print $0} ' > NS1_20180715_US_HALF-CLEANED.csv


# Deduplicate ticks list before comparing them from each datasets

# For Sharadar SF1
# 1. Keep only MRQ (aka MOST-RECENT Quarterly report )
# 2. Count the number of report by ticks.
# 3. Filter report count == 21 (max from initial dataset requested time range; for the moment ...; can be calculated in the future). We need this because we want to maximise datapoint from each tick.
cat SHARADAR_SF1_b4f396bf12b7322892a876eb11353fb7.csv | grep ",MRQ," |  awk 'FS="," {print $1}'  | sort | uniq -c | grep 21 | awk  '{print $2}'  > sf1.ticks.tmp

# For Finsents
# Simply get deduplicated tick list. Using half cleaned list, we get faster processing.
awk  'FS="," {print $1}' NS1_20180715_US_HALF-CLEANED.csv | sort | uniq  > ns1.ticks.tmp


echo "SF1 has $(wc -l sf1.ticks.tmp)\t ticks"
echo "NS1 has $(wc -l ns1.ticks.tmp)\t ticks"

# Keep only deduplicated ones
cat sf1.ticks.tmp  ns1.ticks.tmp | sort | uniq -d > ticks2keep.txt

# Remove temporary files
rm  sf1.ticks.tmp ns1.ticks.tmp

echo "ticks to keep is in this file : ticks2keep.txt"
echo "ticks count found on both files SF1|NS1 : $(wc -l ticks2keep.txt)"

echo -e "### datasets cleanup phase ###\n"

echo "cleanup SHARADAR SF1"
time cat ticks2keep.txt  | xargs -P 10   -I {} egrep --line-buffered  '^{},' SHARADAR_SF1_b4f396bf12b7322892a876eb11353fb7.csv | grep ',MRQ,'  > SF1.cleaned.csv
echo "cleanup SHARADAR SEP"
time cat ticks2keep.txt | xargs -P 10   -I {} egrep --line-buffered  '^{},' SHARADAR_SEP_fb32049b0552692c7ed3619036acb940.csv > SEP.cleaned.csv
echo "cleanup FINSENTS NS1"
time cat ticks2keep.txt | xargs -P 10   -I {} egrep --line-buffered  '^{},' NS1_20180715_US_HALF-CLEANED.csv > NS1.cleaned.csv

echo "on sharadar datasets, add headers provided by originals datasets"
echo -e "$(head -n 1 SHARADAR_SEP_fb32049b0552692c7ed3619036acb940.csv)\n$(cat SEP.cleaned.csv)" > SEP.cleaned.csv
echo -e "$(head -n 1 SHARADAR_SF1_b4f396bf12b7322892a876eb11353fb7.csv)\n$(cat SF1.cleaned.csv)" > SF1.cleaned.csv

