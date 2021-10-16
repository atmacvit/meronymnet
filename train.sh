#train
python Meronymnet/Train/boxGCNVAE.py
python Meronymnet/Train/maskVAE.py

#label-maps-processing
python Meronymnet/arch/label2obj/process_generation/datasets/convert.py

#train c-spade
bash Meronymnet/Train/c-spade_train.sh
