import torch
import os
import json
import numpy as np 
import pandas as pd 
import PIL
from torchvision import transforms

def _process_input(location):
    img_array = PIL.Image.open(location + '/inputfile.png')
    tt = transforms.ToTensor()
    img_t = tt(img_array)
    feature_list = img_t.unsqueeze(0)
    if not os.path.isdir('/curate/output/derived'):
        os.mkdir('/curate/output/derived')
    np.save('/curate/output/derived/processed.npy', feature_list)

    return {}

def _process_result(location):
    return {}