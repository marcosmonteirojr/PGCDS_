#!/bin/bash

#while read i;
#do
#	~/anaconda3/envs/tese2/bin/python ./GA.py $i
#done < ./Bases/bases5.txt

while read i;
do
	~/anaconda3/envs/tese2/bin/python ./Selecao_dinanica_media_desvio.py $i
done < ./tt/Bases/bases3.txt
