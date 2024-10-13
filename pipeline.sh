#!/usr/bin/env bash
SECONDS=0
python 1_find_concepts.py
#python 2_map.py
python 2_rag.py
python 3_map.py
python 4_reduce.py
duration=$SECONDS
echo "$((duration / 60)) minutes and $((duration % 60)) seconds elapsed."