import numpy as np
import json
import matplotlib.pyplot as plt
import requests 
import tqdm

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


# Download files

img_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar"
annot_url = "https://cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz"

save_path_img = "/img_zip"
save_path_annot = "/annot_zip"

download_url(img_url,save_path_img)
download_url(annot_url,save_path_annot)


#


