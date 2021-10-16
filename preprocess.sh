#!/bin/bash

for object_name in 'bicycle' 'sheep' 'bird' 'motorbike' 'person' 'dog' 'horse' 'cow' 'aeroplane' 'cat'
do
   python Dataset/dump_data.py object_name
   python Dataset/dump_images.py object_name
done

python Dataset/preprocessRawData.py
