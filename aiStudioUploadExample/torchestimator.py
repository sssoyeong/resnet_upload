'''
MIT License

Copyright (c) [2020] [Seokkeun Yi]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import subprocess
import sys
import os
import pickle
import shutil
# for handling json
import json
import math
# for monitoring
from time import sleep
from lrcurve import PlotLearningCurve
# for get model
import torch
# to make datetime
from datetime import datetime
import requests
import time
import ast
import astunparse

MAX_DELAY_TIME = 600.0 # seconds => 600.0s = 10mins

def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            os.chmod(dir, 0o777)
    except OSError:
        print("Error: Failed to create the directory")

class TorchEstimator:
    def __init__(self,model_name="",net_name="",script_params={}):
        self.apiurl = 'https://kportal.cirn.re.kr:8443'
        self.netdataloader_name = 'netdataloader'
        self.netdataloader_filename = self.netdataloader_name + '.py'
        self.network_filename = 'net.py'
        self.torchnet_filename = 'torchmodel.py'
        self.torchmodel_filename = 'torchmodel.pth'
        self.model_name = model_name
        self.net_name = net_name
        self.script_params = script_params
        self.init_params()
        self.init_classes()
        
    def init_params(self):
        # Handling Arguments
        # Number of processors to run MPI
        nprocs = 1
        if 'nprocs' in self.script_params:
            nprocs = self.script_params['nprocs']
        self.nprocs = nprocs
        if 'debug' in self.script_params:
            self.debug = True
        else:
            self.debug = False
        if 'validation' in self.script_params:
            self.validation = True
        else:
            self.validation = False
        # Make Arguments List
        args = []
        for key, value in self.script_params.items():
            if key == "no-cuda" and value == False:
                pass
            elif key == "debug" and value == False:
                self.debug = False                
            elif key == "validation" and value == False:                
                self.validation = False                
            elif key == "no-evaluation" and value == False:
                pass
            else:
                args.append('--'+key) # names of arguments
            if key != 'debug' and key != 'no-cuda' and key != 'validation' and key != 'no-evaluation':
                args.append(str(value))
                
        # Problem Type
        if 'problem-type' in self.script_params:
            self.problem_type = self.script_params['problem-type']
        
        # Initialize train params
        self.training = False
        self.trained = False
        
        ########## PATH ##########
        ## Estimator's Directory #
        # Get path for this module
        self.this_path = os.path.dirname(os.path.abspath(__file__))
        ##########################
        # Get workspace path
        ##########################                
        #EDISON_SCIDATA_PATH = '/'
        notebook_path = os.getcwd()
        if notebook_path.startswith('./') :
            notebook_path = notebook_path[5:]
            #notebook_path = EDISON_SCIDATA_PATH + notebook_path[5:]
        if self.debug:
            print("Notebook Path: ", notebook_path)
        pos1 = notebook_path.find('workspace')
        if pos1 == -1:
            print("Wrong workspace path")
            return
        pos2 = notebook_path[pos1+10:].find('/')
        WORKSPACE_PATH = ''
        if pos2 == -1:
            print("Workspace path was found.")
            WORKSPACE_PATH = notebook_path
            self.workspace_name = notebook_path[pos1+10:]
        else:
            print("Warning: Path of the notebook file must be located at the root of the workspace folder.")
            WORKSPACE_PATH = notebook_path[:pos1+10+pos2] # workspace is 9 letters   
            self.workspace_name = notebook_path[pos1+10:pos1+10+pos2]
        self.workspace_path = WORKSPACE_PATH
        # get user id from workspace path
        self.user_id = WORKSPACE_PATH[6:6+(WORKSPACE_PATH[6:].find('/'))]
        self.home_path = EDISON_SCIDATA_PATH + '/' + self.user_id
        # Job path settings (mkdir)
#         JOB_PATH = WORKSPACE_PATH + '/job/job-' + str(JOB_INDEX)        
        JOB_PATH = WORKSPACE_PATH + './job'
        if self.debug:
            print("Job Path: ", JOB_PATH)
        createDirectory(JOB_PATH)
        self.job_path = JOB_PATH
        self.has_job = False
        self.job_title = ""
        # 5 is after 'home/'        
        self.real_output_path = EDISON_SCIDATA_PATH + JOB_PATH[5:]
        self.real_workspace_path = EDISON_SCIDATA_PATH + WORKSPACE_PATH[5:]
        # Add Model Path
        if self.model_name != "":
            MODEL_PATH = WORKSPACE_PATH + './model/' + self.model_name
            args.append('--model-path')
            args.append(MODEL_PATH)
            self.script_params['model-path'] = MODEL_PATH
            self.model_path = MODEL_PATH
            self.trained = True
        # Add Network Name
        elif self.net_name != "":            
            args.append('--net-name')
            args.append(self.net_name)
            self.script_params['net-name'] = self.net_name
        # Verify Arguments
        if self.debug:
            print(args)            
        self.args = args
        
    def init_classes(self):
        self.class_object = None
        self.net_object = []
        temp_dir = './a'
        createDirectory(temp_dir)
        nbfilename = self._get_nbname()
        nbname = nbfilename.split(".ipynb")[0]
        self._save_this_nb_to_py(nbfilename,dest_dir=temp_dir)
        pyfile = temp_dir + '/' + nbname + '.py'
        has_net = False
        has_dloader = False        
        with open(pyfile) as f:
            ast_pyfile = ast.parse(f.read())
            for node in ast_pyfile.body[:]:
                if type(node) == ast.ClassDef:
                    if node.name == 'Net' and has_net==False:
                        print("A neural network definition has been found.")
                        has_net = True
                        self.net_object.append(node)
                    elif node.name == 'DatasetLoader' and has_dloader==False:
                        print("A dataset loader definition has been found.")
                        has_dloader = True
                    else:
                        ast_pyfile.body.remove(node)
                elif type(node) == ast.ImportFrom:
                    if node.module.find('sdr')!=-1: # Remove Import of SDR Libraries
                        ast_pyfile.body.remove(node)
                    elif not has_net:
                        self.net_object.append(node)
                elif type(node) == ast.Import:            
                    for modulename in node.names:
                        if modulename.name.find('TorchEstimator')!=-1: # Remove Import of Torch Estimator
                            ast_pyfile.body.remove(node)
                        elif not has_net:
                            self.net_object.append(node)
                elif type(node) != ast.FunctionDef:
                    ast_pyfile.body.remove(node)

            if len(ast_pyfile.body)<1:
                print("Warning: Cannot analyze the notebook file.")
#                 raise ValueError("Cannot analyze the notebook file.")
        
        self.class_object = ast_pyfile
        if not has_net:
            print("Warning: A neural network definition was not found.")
        if not has_dloader:
            print("Warning: A dataset loader definition was not found.")
            
    def make_classes(self):
        if self.has_job:
            # Network and Dataloader
            with open(self.this_job_path + '/' + self.netdataloader_filename,"w") as f:
                f.write(astunparse.unparse(self.class_object))
            # Network only
            with open(self.this_job_path + '/' + self.network_filename,"w") as f:
                f.write(astunparse.unparse(self.net_object))

    def make_job_path(self):
        timenow = datetime.now().strftime('%Y%m%d%H%M%S')
        self.job_num = timenow
        self.job_location = 'job-' + timenow
        if self.job_title == "":
            self.job_title = self.job_location
        self.this_job_path = self.job_path + '/' + self.job_location        
        self.job_script = self.this_job_path + '/job.sh'
        self.output_path = self.real_output_path + '/' +  self.job_location
#         if not os.path.isdir(self.this_job_path):
#             os.mkdir(self.this_job_path)
        createDirectory(self.this_job_path)
        self.has_job = True
        return self.this_job_path
        
    def make_shell_script(self,argstr):                
        with open(self.this_path+"/eqspec.json", "r") as eqspec_json:
            eqspec = json.load(eqspec_json)
        num_cores_cpu = eqspec['num-cores-cpu']
        num_cores_gpu = eqspec['num-cores-gpu']
        max_nodes = eqspec['max-nodes']
        
        #calculate max procs
        max_procs_cpu = num_cores_cpu * max_nodes
        max_procs_gpu = num_cores_gpu * max_nodes
        
        if 'no-cuda' in self.script_params:
            # use cpu
            if self.script_params['no-cuda']==True:
                ntasks_per_node = num_cores_cpu
                if self.nprocs > max_procs_cpu:
                    print("The maximum number of cores(CPU) has been exceeded.")
                    return "ERROR"
            else:
                # use gpu
                ntasks_per_node = num_cores_gpu
                if self.nprocs > max_procs_gpu:
                    print("The maximum number of cores(GPU) has been exceeded.")
                    return "ERROR"  
        else:
            # use gpu
            ntasks_per_node = num_cores_gpu
            if self.nprocs > max_procs_gpu:
                print("The maximum number of cores(GPU) has been exceeded.")
                return "ERROR"
            
        
        # set ntasks per node
        ntasks = self.nprocs
        if ntasks == 1:
            ntasks_per_node = 1        
        elif ntasks <= 0:
            print("A problem was found with the parallel parameters.")
            return "ERROR"
        
        # calculate and nnodes        
        nnodes = math.ceil(ntasks/ntasks_per_node)
        self.nnodes = nnodes
        
        shell_script='''\
#!/bin/bash
#SBATCH --job-name=job-{}
#SBATCH --output={}/std.out
#SBATCH --error={}/std.err
#SBATCH --nodes={}
#SBATCH --ntasks={}
#SBATCH --ntasks-per-node={}
#SBATCH --exclusive

HOME={}
JOBDIR={}
curl {}/api/jsonws/SDR_base-portlet.dejob/studio-update-status -d deJobId={} -d Status=RUNNING 
conda activate torch
/usr/local/bin/mpirun -np {} -x TORCH_HOME=${{HOME}} -x PATH -x HOROVOD_MPI_THREADS_DISABLE=1 -x NCCL_SOCKET_IFNAME=^docker0,lo -mca btl_tcp_if_exclude lo,docker0  -mca pml ob1 singularity exec --nv -H ${{HOME}}:${{HOME}} --nv --pwd ${{JOBDIR}} /data/sdr/singularity-images/userenv3 python ${{JOBDIR}}/train.py {} || error_code=$?

if [ ! "${{error_code}}" = "" ]; then
    echo ${{error_code}}
    echo "failed" > ${{JOBDIR}}/status
    curl {}/api/jsonws/sdr.dejob/studio-update-status -d deJobId={} -d Status=FAILED 
else
    echo ${{error_code}}
    echo "finished" > ${{JOBDIR}}/status
    curl {}/api/jsonws/sdr.dejob/studio-update-status -d deJobId={} -d Status=SUCCESS 
fi
'''.format(self.job_num, self.output_path, self.output_path, nnodes, ntasks, ntasks_per_node, self.home_path, self.this_job_path, self.apiurl, self.job_id, str(self.nprocs), argstr, self.apiurl, self.job_id, self.apiurl, self.job_id)
        
        # FOR LOCAL TEST
#         shell_script='''\
# #!/bin/sh
# horovodrun -np {} python {} {}
# '''.format(str(self.nprocs), self.this_job_path+'/train.py', argstr)
        
        return shell_script

    def get_job_path(self):
        if self.has_job:
            return self.this_job_path
        else:
            print("A working job directory has not been created yet.")

    # job_type = 1 : AI Training by Template, 2 : Normal Script
    def write_metadata(self,job_type=1):
        
        metadata = {}
        metadata['job-type'] = job_type
        metadata['information'] = {}
        # default processor type
        metadata['information']['proc_type'] = "GPU"
        # additional meta data
        metadata['information']['nnodes'] = self.nnodes
        
        if job_type == 1:
            # Example of script_params
            '''
            script_params = {
                'epochs':5,
                'batch-size':64,
                'test-batch-size':128,
                'lr':0.01,
                'momentum':0.5,
                'seed':42,
                'log-interval':10,
                #'no-cuda':False,
                'nprocs':1,
                'loss':'cross_entropy',
                #'loss':'nll_loss',
                'optimizer':'SGD',
                'validation': True,
                'debug': True
            }
            '''            
            
            metadata['hyperparameters'] = {}
            metadata['otherparameters'] = {}
            metadata['others'] = {}

            for key, value in self.script_params.items():
                if key =="problem-type" or key =="nprocs" or key =="debug":
                    metadata['information'][key] = value
                elif key =="net-name" or key =="validation":
                    metadata['information'][key] = value
                elif key == "no-cuda":                
                    if value == True: # use cpu
                        metadata['information']['proc_type'] = "CPU"
                elif key == "epochs" or key == "batch-size" or key =="test-batch-size" or key == "seed" or \
                    key == "lr" or key == "momentum" or key == "loss" or key == "optimizer":
                    metadata['hyperparameters'][key] = value
                elif key =="log-interval":
                    metadata['otherparameters'][key] = value
                else:
                    metadata['others'][key] = value
                    
        elif job_type == 2:            
            metadata['parameters'] = {}            
            for key, value in self.script_params.items():
                if key =="problem-type" or key =="nprocs" or key =="debug":
                    metadata['information'][key] = value
                elif key == "no-cuda":
                    if value == True: # use cpu
                        metadata['information']['proc_type'] = "CPU"
                else:
                    metadata['otherparameters'][key] = value
    

        with open(self.this_job_path + "/meta-job.json", "w") as json_file:
            json.dump(metadata, json_file)
        
    def copy_train_script(self):
        if self.has_job:
            # copy train.py to job path
            org_train_script_path = self.this_path + '/train.py'
            train_script_path = self.this_job_path + '/train.py'
            shutil.copy(org_train_script_path, train_script_path)
    
    def set_job_title(self,job_title=""):
        ##### Set Job Title #####
        if job_title != "":
            job_title = job_title.replace(" ","_") # Replace spaces with hyphens
            self.job_title = job_title
        else:
            self.job_title = ""
        
    
    def fit(self,job_title="",input_data=None,input_labels=None):
        # make arguments list to one string
        argstr = ' '.join(self.args)
        ##### Set Job Title #####
        self.set_job_title(job_title)
        ##### Make dir for new job #####
        self.make_job_path()
        ##### Make Net Class File and DatasetLoader Class File #####
        self.make_classes()
        ##### request submit job (register job to database) - API Call #####
        self._request_submit_job()
        # copy train.py to job path
        self.copy_train_script()
        # Writing Training Script        
        with open(self.job_script,'w') as shfile:
            ##### Make Shell Script #####
            shell_script=self.make_shell_script(argstr)
            if shell_script == "ERROR":
                # Failed to Predict
                self._request_update_status("FAILED")
                return
            shfile.write(shell_script)
        # Set permission to run the script
        os.chmod(self.job_script, 0o777)
        # Save Dataset
        if input_data is not None and input_labels is not None:
            with open(self.this_job_path+'/dataset.pkl', 'wb') as f:
                pickle.dump((input_data, input_labels), f, pickle.HIGHEST_PROTOCOL)
        ##### Write Meta Data JSON File #####
        self.write_metadata()
        # Request Job Submission
        self.training = self._request_to_portal()
        
    def create_model_metadata(self,model_name,model_path,score):
        metadata = {}
        emptylist = []
        metadata['modelName'] = model_name
        metadata['jobFrom'] = 'AI Studio API'
        metadata['framework'] = 'PyTorch'
        metadata['parentJob'] = self.job_id
        metadata['studyName'] = ""
        metadata['inputs'] = emptylist
        metadata['trialNumber'] = -1
        metadata['accuracy'] = float(score)
        metadata['jobType'] = 81
        timenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metadata['createDate'] = timenow
        org_model_metadata_path = self.this_job_path + '/modelmeta.json'
        with open(org_model_metadata_path, "w") as json_file:
            json.dump(metadata, json_file)
        model_metadata_path = model_path + '/modelmeta.json'
        if os.path.exists(org_model_metadata_path):
            shutil.copy(org_model_metadata_path, model_metadata_path)
        
    def register_model(self, model_name):
        if not self.trained:
            if os.isfile(self.this_job_path + '/' + self.torchmodel_filename):
                self.trained=True
            else:
                print("Error: No model trained.")
                return
        # add model information to database(file db or web db)
        if self.trained:
            # create model folder        
            model_root_path = self.workspace_path + '/model'
            model_path = model_root_path + '/' + model_name
            createDirectory(model_root_path)
            createDirectory(model_path)
            if self.net_name is not "":
                # copy network file to model path
                # $WORKSPACE/nets{net_name} -> $WORKSPACE/model/{model_name}/torchmodel.py
                org_net_path = self.workspace_path + '/net/' + self.net_name + '.py'
                net_path = model_path + '/' + self.torchnet_filename
                shutil.copy(org_net_path, net_path)
            else:
                org_net_path = self.this_job_path + '/' + self.network_filename
                net_path = model_path + '/' + self.torchnet_filename
                shutil.copy(org_net_path, net_path)
            # copy model file to model path
            # $JOB_PATH/torchmodel.pth -> $WORKSPACE/model/{model_name}/torchmodel.pth
            org_modelfile_path = self.this_job_path + '/' + self.torchmodel_filename
            modelfile_path = model_path + '/' + self.torchmodel_filename
            shutil.copy(org_modelfile_path, modelfile_path)
            # copy service.json to model path            
            org_service_file_path = self.this_job_path + '/service.json'
            service_file_path = model_path + '/service.json'
            if os.path.exists(org_service_file_path):
                shutil.copy(org_service_file_path, service_file_path)
            # copy result graph to model path
            if self.problem_type == "classification":
                org_graph_file_path = self.this_job_path + '/confusionMatrix.png'
                graph_file_path = model_path + '/confusionMatrix.png'
            if self.problem_type == "regression":
                org_graph_file_path = self.this_job_path + '/regressionAccuracy.png'
                graph_file_path = model_path + '/regressionAccuracy.png'
            if os.path.exists(org_graph_file_path):
                shutil.copy(org_graph_file_path, graph_file_path)
            # copy score to model path
            org_score_path = self.this_job_path + '/score'
            with open(org_score_path,"r") as score_file:
                score = score_file.readline()
                pos = score.find(':')
                score = score[pos+2:]
            score_path = model_path + '/score'
            if os.path.exists(org_score_path):
                shutil.copy(org_score_path, score_path)
            self.create_model_metadata(model_name,model_path,score)
            
    
#     def extract_network(self):
#         if self.has_job:
#             # Open netdataloader python file by ast
#             pyfile = self.this_job_path + '/' + 'netdataloader.py'
#             has_net = False
#             # Extract Net class in the .py file and Save to 'net' Directory
#             with open(pyfile) as f:
#                 ast_pyfile = ast.parse(f.read())
#                 for node in ast_pyfile.body[:]:
#                     if type(node) == ast.ClassDef:
#                         if node.name == 'Net':
#                             print("A neural network definition has been found.")
#                             has_net = True
#                         else:
#                             ast_pyfile.body.remove(node)
#                     elif type(node) == ast.ImportFrom:
#                         if node.module.find('sdr')!=-1: # Remove Import of SDR Libraries
#                             ast_pyfile.body.remove(node)
#                     elif type(node) == ast.Import:            
#                         for modulename in node.names:
#                             if modulename.name.find('TorchEstimator')!=-1: # Remove Import of Torch Estimator
#                                 ast_pyfile.body.remove(node)
#                     else:
#                         ast_pyfile.body.remove(node)
                
#                 if len(ast_pyfile.body)<1:
#                     raise ValueError("Cannot analyze the netdataloader file.")

#             if not has_net:
#                 print("Error: A neural network definition was not found.")            
#             with open(self.this_job_path + '/net.py',"w") as f:
#                 f.write(astunparse.unparse(ast_pyfile))
    
        
    def monitor(self, timeout=MAX_DELAY_TIME):        
        if 'epochs' in self.script_params:
            self.epochs = self.script_params['epochs']
        # for end condition
        epochs = int(self.epochs)
        
        if self.validation:
            plot = PlotLearningCurve(
                mappings = {
                    'loss': { 'line': 'train', 'facet': 'loss' },
                    'val_loss': { 'line': 'validation', 'facet': 'loss' },
                    'acc': { 'line': 'train', 'facet': 'acc' },
                    'val_acc': { 'line': 'validation', 'facet': 'acc' }
                },
                facet_config = {
                    'loss': { 'name': 'Loss', 'limit': [None, None], 'scale': 'linear' },
                    'acc': { 'name': 'Accuracy', 'limit': [None, 1], 'scale': 'linear' }
                },
                xaxis_config = { 'name': 'Epoch', 'limit': [0, epochs] }
            )
        else:
            plot = PlotLearningCurve(
                mappings = {
                    'loss': { 'line': 'train', 'facet': 'loss' },
                    'acc': { 'line': 'train', 'facet': 'acc' }
                },
                facet_config = {
                    'loss': { 'name': 'Loss', 'limit': [None, None], 'scale': 'linear' },
                    'acc': { 'name': 'Accuracy', 'limit': [0, 1], 'scale': 'linear' }
                },
                xaxis_config = { 'name': 'Epoch', 'limit': [0, epochs] }
            )
        # log monitoring loop
        delay_time = 0.
        try:
            with open(self.this_job_path + "/epoch.log","r") as f:
                while True:
                    where = f.tell()
                    line = f.readline().strip()
                    if not line:
                        sleep(0.1)
                        delay_time += 0.1
                        f.seek(where)
                        if timeout is not None:
                            if delay_time > timeout:
                                print("Delay has been exceeded.")
                                break
                        else:
                            if delay_time > MAX_DELAY_TIME:
                                print("Delay has been exceeded.")
                                break                    
                    else:
                        delay_time = 0. # reset delay time
                        # print(line) # already has newline
                        strlist = line.split(',')
                        phase = strlist[0].strip()
                        epoch = strlist[1].strip()
                        loss = strlist[2].strip()
                        acc = strlist[3].strip()
                        print("Phase: {}, Epoch: {}, Loss: {}, Acc: {}".format(phase,epoch,loss,acc))
                        # append and update
                        if phase=="Train":
                            plot.append(epoch, {
                                'loss': loss,
                                'acc': acc
                            })
                            plot.draw()
                        else:
                            plot.append(epoch, {
                                'val_loss': loss,
                                'val_acc': acc
                            })
                            plot.draw()
                        # End Condition (If the Last Epoch Finished, Terminate the Loop)
                        if self.validation:
                            if phase=="Validation":
                                if int(epoch) == epochs:
                                    break
                        else:
                            if int(epoch) == epochs:
                                break 
        except:
            print("Even one epoch has not been completed.\nPlease execute the command again after a while.")        
                        
        
    def predict(self,job_title="",dataset_loader=""):
        argstr = ' '.join(self.args)
        argstr = argstr + ' --prediction'
        if self.has_job==False:
            ##### Set Job Title #####
            self.set_job_title(job_title)
            ##### Make dir for new job #####
            self.make_job_path()
            ##### Make Net Class File and DatasetLoader Class File #####
            self.make_classes()
            # copy train.py to job path
            self.copy_train_script()
        else:
            if job_title!="":
                print("Warning: Prediction will be performed within the already trained task. The entered job title is ignored.")
        ##### request submit job (register job to database) - API Call #####
        self._request_submit_job()
        # FOR LOCAL TEST
        with open(self.job_script,'w') as shfile:
#                 shell_script='''\
# #!/bin/sh
# horovodrun -np {} python {} {}
# '''.format(str(self.nprocs),self.job_path+'/train.py',argstr)
            ##### Make Shell Script #####
            shell_script=self.make_shell_script(argstr)
            if shell_script == "ERROR":
                # Failed to Predict
                return
            shfile.write(shell_script)
        # Set permission to run the script
        os.chmod(self.job_script, 0o777)
        ##### Write Meta Data JSON File #####
        self.write_metadata()
        # Request Job Submission
        self.training = self._request_to_portal()
    
    # Report for Prediction
    def report(self):
        try:
            with open(self.this_job_path + "/epoch_prediction.log","r") as f:
                line = f.readline()
                print(line)
        except:
            print("Prediction has not been completed.\nPlease execute the command again after a while.")        
        
    def get_model(self):
        if not self.trained:
            if os.isfile(self.this_job_path + '/' + self.torchmodel_filename):
                self.trained=True
        # Load Network
        if self.trained:
            if self.net_name is not "":
                print("Network was found.")
                # set system path to load model
                modulename = self.net_name
                # net path e.g.) $HOME/workspace/ws-1/net
                netpath = os.path.join(self.workspace_path, 'net')        
                sys.path.append(netpath)
                # Custom Model
                import importlib
                torchnet = importlib.import_module(modulename)
                # set model
                model = torchnet.Net()                
            else:                      
                sys.path.append(self.this_job_path)
                import importlib
                netdataloader = importlib.import_module(self.netdataloader_name)
                model = netdataloader.Net()
            # Load weights of the model from .pth file
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(self.this_job_path + '/' + self.torchmodel_filename))
            else:
                device = torch.device('cpu')
                model.load_state_dict(torch.load(self.this_job_path + '/' + self.torchmodel_filename,map_location=device))
            return model
        else:
            print("Training has not been completed.")
            return None
    
    def submit(self,script_path,job_title=""):
        # make arguments list to one string
        argstr = ' '.join(self.args)
        ##### Set Job Title #####
        self.set_job_title(job_title)
        ##### Make dir for new job #####
        self.make_job_path()
        ##### request submit job (register job to database) - API Call #####
        self._request_submit_job()
        # copy training script to job path
        if self.has_job:
            # copy training script as train.py to job path            
            train_script_path = self.this_job_path + '/train.py'
            shutil.copy(script_path, train_script_path)
        # Writing Training Script        
        with open(self.job_script,'w') as shfile:
            ##### Make Shell Script #####
            shell_script=self.make_shell_script("")
            if shell_script == "ERROR":
                # Failed to Predict
                return
            shfile.write(shell_script)
        # Set permission to run the script
        os.chmod(self.job_script, 0o777)
        ##### Write Meta Data JSON File #####
        self.write_metadata(2) # type 2
        # Request Job Submission
        self.training = self._request_to_portal()
    
    def _request_submit_job(self):
        data = {
          'screenName': self.user_id,
          'title': self.job_title,
          'targetType': '81', # targetType 81 is for normal ai job(train,predict)
          'workspaceName': self.workspace_name,
          'location': self.output_path # real job path
        }
        if self.debug:
            print(data)        
        response = requests.post(self.apiurl+'/api/jsonws/sdr.dejob/studio-submit-de-job', data=data)
        if response.status_code == 200:            
            self.job_id = response.json()            
            print("Job ID-{} was submitted.".format(self.job_id))
        else:
            print("A problem occured when generating the job.")
        print("Job was generated in database.")
    
    def _run_slurm_script(self):
        data = {
          'screenName': self.user_id,
          'location': self.output_path
        }
        if self.debug:
            print(data)
        print("Running Slurm script...")
        response = requests.post(self.apiurl+'/api/jsonws/sdr.dejob/slurm-de-job-run', data=data)
        # waiting for slurm job id
        time.sleep(3)
        try:
            with open(self.this_job_path+'/job.id','r') as f:
                idstr = f.readline()
                idstr = idstr.strip()
                print("Batch job ID-{} is running on the HPC.".format(idstr))
                self.slurm_job_id = int(idstr)
                # Moved to the job.sh
#                 self._request_update_status("RUNNING")
                self.trained = True
        except:
            print("The requested training job has failed.")
        
    def status(self):
        self._request_get_status()
        
    def _request_update_status(self,status):        
        try:
            data = {
              'deJobId': self.job_id,
              'Status': status
            }
            if status == "SUCCESS": # Already finished, Don't change the status
                return
            if status == "FAILED": # Already finished, Don't change the status
                return
            response = requests.post(self.apiurl+'/api/jsonws/sdr.dejob/studio-update-status', data=data)
            if self.debug:
                if response.status_code == 200:
                    print("Job status({}) has been updated.".format(status))
        except:
            print("Error: Slurm Job Not Found.")
        
        
    def _request_get_status(self):
        print("Getting Status of Requested Job on the Portal.")
        try:
            data = {
              'deJobId': self.job_id
            }
            response = requests.post(self.apiurl+'/api/jsonws/sdr.dejob/get-de-job-data', data=data)
            if response.status_code == 200:
                resjson = response.json()
                print('--------------------------------')
                print('Job ID: {}'.format(resjson['deJobId']))
                print('Job Title: {}'.format(resjson['title']))
                print('Job Directory: {}'.format(self.this_job_path))                
                print('Start Date: {}'.format(resjson['startDt']))
                print('End Date: {}'.format(resjson['endDt']))
                print('--------------------------------')
                print('Status: {}'.format(resjson['status']))
            else:
                print("Error: Getting status of the job has failed.")
        except:
            print("Error: Running Job Not Found.")
    
    def cancel(self):
        self._request_to_portal_cancel_job()
        
    def _request_to_portal(self):
        print("Job Requested to Portal.")
        # FOR LOCAL TEST
        # lrcurve & pytorch crash
#         os.environ['MKL_THREADING_LAYER'] = 'GNU'        
#         proc = subprocess.Popen(self.job_script,
#                                 universal_newlines=True, # Good Expression for New Lines                                
#                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#                                 )
#         try:
#             outs, errs = proc.communicate(timeout=15)
#         except subprocess.TimeoutExpired:
#             proc.kill()
#             outs, errs = proc.communicate()
#         print(outs)        
        # Request to Portal to call slurm script
        self._run_slurm_script()
        # if there is no error
        return True
    
    def _request_get_status_return(self):
        try:
            data = {
              'deJobId': self.job_id
            }
            response = requests.post(self.apiurl+'/api/jsonws/sdr.dejob/get-de-job-data', data=data)
            if response.status_code == 200:
                resjson = response.json()
                status = resjson['status']                
                return status
            else:
                print("Error: Getting status of the job has failed.")
                return "ERROR"
        except:
            print("Error: Running Job Not Found.")
            return "ERROR"
        
    
    def _request_to_portal_cancel_job(self):         
        status = self._request_get_status_return()
        if status == "SUCCESS":
            print("The job has already been successfully finished.")
            return
        
        if status == "FAILED":
            print("The job has already been ended.")
            return
        
        print("Canceling a Requested Job on the Portal.")
        try:
            data = {
              'jobId': self.slurm_job_id,
              'screenName': self.user_id
            }
            response = requests.post(self.apiurl+'/api/jsonws/sdr.dejob/slurm-de-job-cancel', data=data)
            if response.status_code == 200:
                print("The job was successfully canceled.")
                self._request_update_status("CANCELLED")
        except:
            print("Error: Slurm Job Not Found.")
            
    def _get_nbname(self):
        from notebook import notebookapp
        import urllib
        import json
        import os
        import ipykernel
        connection_file = os.path.basename(ipykernel.get_connection_file())
        kernel_id = connection_file.split('-', 1)[1].split('.')[0]
        import urllib.request
        opener = urllib.request.build_opener()
        import os
        opener.addheaders = [('Authorization', 'Token '+os.environ.get('JPY_API_TOKEN'))]
        req = opener.open('http://localhost:8888/sdr/user/'+self.user_id+'/api/sessions')
        raw_data = req.read()
        data = json.loads(raw_data.decode('utf-8'))
        for each in data:
            if each['kernel']['id'] == kernel_id:
                nbname = each['notebook']['name']
        return nbname
    
    def _save_this_nb_to_py(self,nbname,dest_dir="./"):
        import subprocess
        filepath = os.getcwd()+os.sep+nbname        
        ipynbfilename=nbname
        try:
            #!jupyter nbconvert --to script {filepath} --output-dir={dest_dir}
            subprocess.check_output("jupyter nbconvert --to script "+filepath+" --output-dir="+dest_dir, shell=True)
            return dest_dir+os.sep+ipynbfilename.split(".ipynb")[0]+'.py'
        except:
            raise ValueError(".py cannot be generated via current notebook.")
        return False
