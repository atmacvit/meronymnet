#infer
python Meronymnet/Inference/boxGCNVAE.py
python Meronymnet/Inference/maskVAE.py

#label-maps-processing
python Meronymnet/arch/label2obj/process_generation/convert.py

sh Meronymnet/arch/label2obj/c-spade_test.sh
