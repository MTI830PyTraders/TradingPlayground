#!/bin/bash

# NS1 cleanup
# Remove NS1 ticks without data
# Keep only US ticks
egrep '^(\w){1,8}_US'  NS1_20180715.csv | awk -F ',' '$6!="0.0" {print $0} ' > NS1_20180715_US_CLEANED.csv

# Deduplicate ticks before comparing datasets
awk  'FS="," {print $1}'  SHARADAR_SF1_b4f396bf12b7322892a876eb11353fb7.csv | sort | uniq > sf1.tmp


awk  'FS="," {print $1}' NS1_20180715_US_CLEANED.csv | sort | uniq  > ns1.tmp


echo "SF1 has $(wc -l sf1.tmp) ticks"
echo "NS1 has $(wc -l  ns1.tmp) ticks"

# Keep only deduplicated ones
cat sf1.tmp  ns1.tmp | sort | uniq -d > ticks2keep.txt

rm  sf1.tmp ns1.tmp


echo "ticks to keep is in this file : ticks2keep.txt"
echo "ticks count found on both files SF1|NS1 : $(wc -l ticks2keep.txt)"
