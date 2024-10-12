#!/usr/bin/env bash
SECONDS=0
python 1_find_concepts.py
if [ $?!=0 ];
then
    echo "Error in 1_find_concepts.py"
fi
python 2_map.py
if [ $?!=0 ];
then
    echo "Error in 2_map.py"
fi
python 3_map.py
if [ $?!=0 ];
then
    echo "Error in 3_map.py"
fi
python 4_reduce.py
if [ $?!=0 ];
then
    echo "Error in 4_reduce.py"
fi
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."