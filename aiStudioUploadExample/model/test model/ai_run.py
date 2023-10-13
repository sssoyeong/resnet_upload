import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd 
import os

class DatasetLoader():
    def __init__(self, dataset_path='./', run_service=False):
        
        ###########################
        # define processing logic #
        ###########################
        if run_service:
            
            self.processed_data = []
            
        else:
            train_dataset, valid_dataset, test_dataset = [], [], []

            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
            self.test_dataset = test_dataset


    def get_train_dataset(self, validation=True):
        if validation is True:
            return (self.train_dataset, self.valid_dataset)
        else:
            return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_processed_data(self):
        return self.processed_data

def _service(input_dir, model_dir):
    # load service.json 
    with open(model_dir + 'service.json') as json_file:
        json_data = json.load(json_file)

    # numpy or csv or txt file as input
    if os.path.isfile(input_dir + '/derived/processed.npy'):
        processedX = np.load(input_dir + '/derived/processed.npy')
    elif os.path.isfile(input_dir + '/metadata/dm.json'):
        with open(input_dir + '/metadata/dm.json') as input_file:
            input_data = json.load(input_file)
        try:
            input_df = pd.DataFrame.from_dict(input_data)
        except:
            input_df = pd.DataFrame.from_dict([input_data])
        processedX = torch.from_numpy(input_df.values).float()
        #varX = torch.utils.data.DataLoader(varX)
    else:
        # input data to dataset
        dataset_loader = DatasetLoader(input_dir, run_service=True)
        processedX = dataset_loader.get_processed_data()

    varX = torch.utils.data.DataLoader(processedX)

    predicted_result = []
    # load model
    models = json_data['model']
    for key in models:
        modelName = models[key]
        model = Net()
        model.load_state_dict(torch.load(model_dir + modelName, map_location = torch.device('cpu')))
        model.eval()

        for i,data in enumerate(varX):
            predicted = model(data)
            if json_data['task'] == "Classification":
                predicted_result.extend(predicted.argmax(dim=1).detach().numpy().tolist())
            else:
                predicted_result.extend(predicted.detach().numpy().tolist())

    # post-processing

    return predicted_result
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd 
import os
import shutil

## copy & paste netdataloader.py
class DatasetLoader():
    def __init__(self, dataset_path='./', run_service=False):
        # define processing 
        if run_service:
            from torchvision import transforms
            import PIL
            from glob import glob
            img_list = glob(dataset_path + '/*.png')
            out_list = []
            for img in img_list:
                img_array = PIL.Image.open(img)
                tt = transforms.ToTensor()
                img_t = tt(img_array)
                out_list.append(img_t.unsqueeze(0))
            #print(img_t)
            self.processed_data = torch.cat(out_list)
            
            source = dataset_path + '/inputfile.csv'
            dest = '/curate/output/derived/predicted.csv'
            shutil.copy(source, dest)
            
        else:
            train_dataset, valid_dataset, test_dataset = [], [], []

            self.train_dataset = train_dataset
            self.valid_dataset = valid_dataset
            self.test_dataset = test_dataset


    def get_train_dataset(self, validation=True):
        if validation is True:
            return (self.train_dataset, self.valid_dataset)
        else:
            return self.train_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_processed_data(self):
        return self.processed_data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


def _service(input_dir, model_dir):
    # load service.json 
    with open(model_dir + 'service.json') as json_file:
        json_data = json.load(json_file)

    # numpy or csv or txt file as input
    if os.path.isfile(input_dir + '/derived/processed.npy'):
        processedX = np.load(input_dir + '/derived/processed.npy')
    elif os.path.isfile(input_dir + '/metadata/dm.json'):
        with open(input_dir + '/metadata/dm.json') as input_file:
            input_data = json.load(input_file)
        try:
            input_df = pd.DataFrame.from_dict(input_data)
        except:
            input_df = pd.DataFrame.from_dict([input_data])
        processedX = torch.from_numpy(input_df.values).float()
        #varX = torch.utils.data.DataLoader(varX)
    else:
        # input data to dataset
        dataset_loader = DatasetLoader(input_dir, run_service=True)
        processedX = dataset_loader.get_processed_data()

    varX = torch.utils.data.DataLoader(processedX)

    predicted_result = []
    # load model
    models = json_data['model']
    for key in models:
        modelName = models[key]
        model = Net()
        model.load_state_dict(torch.load(model_dir + modelName, map_location = torch.device('cpu')))
        model.eval()

        for i,data in enumerate(varX):
            predicted = model(data)
            if json_data['task'] == "Classification":
                predicted_result.extend(predicted.argmax(dim=1).detach().numpy().tolist())
            else:
                predicted_result.extend(predicted.detach().numpy().tolist())

    # post-processing

    return predicted_result