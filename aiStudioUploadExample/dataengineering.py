import sys
import os
# import shutil
# for handling json
import json
import math
# for monitoring
# to make datetime
from datetime import datetime
import requests
import time
import ast
import astunparse

def createDirectory(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            os.chmod(dir, 0o777)
    except OSError:
        print("Error: Failed to create the directory")

class DataEngineering:
    def __init__(self,script_params={}):
        self.apiurl = 'https://sdr.edison.re.kr:8443'
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
        
        # Make Arguments List
        args = []
        for key, value in self.script_params.items():
            if key == "debug" and value == "False":
                pass
            else:
                args.append('--'+key) # names of arguments
            if key != 'debug':                
                args.append(str(value))
        
        ########## PATH ##########
        ## Estimator's Directory #
        # Get path for this module
        self.this_path = os.path.dirname(os.path.abspath(__file__))
        ##########################
        # Get workspace path
        ##########################
        self.edison_sdr_path = '/EDISON/SCIDATA/sdr'
        EDISON_SCIDATA_PATH = '/EDISON/SCIDATA/sdr/draft'
        notebook_path = os.getcwd()
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
        JOB_PATH = WORKSPACE_PATH + '/job'
        if self.debug:
            print("Job Path: ", JOB_PATH)
        createDirectory(JOB_PATH)
        self.job_path = JOB_PATH
        self.has_job = False
        # 5 is after 'home/'        
        self.real_output_path = EDISON_SCIDATA_PATH + JOB_PATH[5:]
        self.real_workspace_path = EDISON_SCIDATA_PATH + WORKSPACE_PATH[5:]
        # Set real path of the data directory
        self.data_dir = os.path.abspath(self.script_params['data_dir'])
        self.real_data_dir = self.replace_dir_to_edison(self.data_dir)
        # Verify Arguments
        if self.debug:
            print(args)            
        self.args = args
        
    def init_classes(self):
        self.class_object = None
        temp_dir = './.temp'
        createDirectory(temp_dir)
        nbfilename = self._get_nbname()
        nbname = nbfilename.split(".ipynb")[0]
        self._save_this_nb_to_py(nbfilename,dest_dir=temp_dir)
        pyfile = temp_dir + '/' + nbname + '.py'
        has_input = False
        has_result = False
        with open(pyfile) as f:
            ast_pyfile = ast.parse(f.read())
            for node in ast_pyfile.body[:]:
                if type(node) == ast.FunctionDef:
                    if node.name == '_process_input':
                        print("A _process_input definition has been found.")
                        has_input = True
                    elif node.name == '_process_result':
                        print("A _process_result definition has been found.")
                        has_result = True
                    else:
                        pass
#                         ast_pyfile.body.remove(node)
                elif type(node) == ast.ImportFrom:
                    if node.module.find('sdr')!=-1: # Remove Import of SDR Libraries
                        ast_pyfile.body.remove(node)
                elif type(node) == ast.Import:            
                    for modulename in node.names:
                        if modulename.name.find('DataEngineering')!=-1: # Remove Import of DataEngineering
                            ast_pyfile.body.remove(node)
                else:
                    ast_pyfile.body.remove(node)

            if len(ast_pyfile.body)<1:
                raise ValueError("Cannot analyze the notebook file.")
        
        self.class_object = ast_pyfile
        if not has_input:
            print("Error: A _process_input definition was not found.")
        if not has_result:
            print("Error: A _process_result definition was not found.")    
        
    def make_classes(self):
        if self.has_job:
            with open(self.this_job_path + '/de_run.py',"w") as f:
                f.write(astunparse.unparse(self.class_object))

    def make_job_path(self):
        timenow = datetime.now().strftime('%Y%m%d%H%M%S')
        self.job_num = timenow
        self.job_location = 'de-' + timenow
        if self.job_title == "":
            self.job_title = self.job_location
        self.this_job_path = self.job_path + '/' + self.job_location        
        self.job_script = self.this_job_path + '/job.sh'
        self.job_list = self.this_job_path + '/job.list'
        self.batch_info = self.this_job_path + '/batch.info'
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
#         max_procs_gpu = num_cores_gpu * max_nodes

        ntasks_per_node = num_cores_cpu
        
        if self.nprocs > max_procs_cpu: # use cpu
            print("The maximum number of cores(CPU) has been exceeded.")
            return "ERROR"          
#         if self.nprocs > max_procs_gpu: # use gpu
#             print("The maximum number of cores has been exceeded.")
#             return "ERROR"
        
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
USER={}
SINDIR=/EDISON/SCIDATA/sdr/singularity-images
source /usr/lib64/anaconda3/etc/profile.d/conda.sh
/usr/local/bin/mpirun -np {} -x PATH ${{SINDIR}}/DE_run ${{SINDIR}}/BatchCurate.jar ${{JOBDIR}} ${{USER}} || error_code=$?
if [ ! "${{error_code}}" = "" ]; then
    echo ${{error_code}}
    echo "failed" > ${{JOBDIR}}/status
    curl {}/api/jsonws/SDR_base-portlet.dejob/studio-update-status -d deJobId={} -d Status=FAILED 
else
    echo ${{error_code}}
    echo "finished" > ${{JOBDIR}}/status
    curl {}/api/jsonws/SDR_base-portlet.dejob/studio-update-status -d deJobId={} -d Status=SUCCESS 
fi
'''.format(self.job_num,self.output_path, self.output_path, nnodes, ntasks, ntasks_per_node, self.home_path, self.output_path, self.user_id, str(self.nprocs), self.apiurl, self.job_id, self.apiurl, self.job_id)
        
        return shell_script
    
    def make_batch_info(self):
        if self.data_dir != "":
            batch_info='''\
pythonCommand:python
singularity.command:singularity
dataset.location:{}
remote.curate:singularity
singularity.Image.path:singularity-images 
docker.command:docker
local.curate:docker
dataset.remote.location:{}
remote.curate.allowed:false
'''.format(self.edison_sdr_path,self.edison_sdr_path)
        else:
            print("Error: Data Dir")
        
        with open(self.batch_info,"w") as f:
            f.write(batch_info)

    def replace_dir_to_edison(self,directory):
        EDISON_SCIDATA_PATH = '/EDISON/SCIDATA/sdr/draft'
        pos1 = directory.find('workspace')
        if pos1 == -1:
            print("Error: Wrong dataset path")
            return
        pos1 = directory.find('/home')
        if pos1 != 0:
            print("Error: Invalid home path")
            return
        subdir = directory[6:]
        return EDISON_SCIDATA_PATH + '/' + subdir
        
        
    def make_job_list(self):
        self.data_dir_list = os.listdir(self.script_params['data_dir'])        
        for directory in self.data_dir_list[:]:
            if directory.find('.') == 0: # filtering cache directory or others like this
                self.data_dir_list.remove(directory)            
        if self.debug:
            print(self.data_dir_list)
        with open(self.job_list,"w") as f:
            for i,directory in enumerate(self.data_dir_list):
                directory = self.real_data_dir + '/' + directory
                f.write("{},-1,0,{},,1,true\n".format(i,directory))
        self.splitDataset(self.this_job_path,self.nprocs)
    
    def splitDataset(self,location, num):
        rf = open(os.path.join(location, 'job.list') , 'r', encoding='utf-8' ) 
        wfl = []
        for idx in range(0,num) :
            fn = os.path.join(location, 'dataset.list_%s' % idx)
            wf = open(os.path.join(location, fn) , 'w', encoding='utf-8' ) 
            wfl.append(wf)

        idx = 0
        while True:
            line = rf.readline()
            if not line:
                break
            wfl[idx].write(line)
            #wfl[idx].write("/")

            idx = idx + 1
            if idx >= num :
                idx = 0

        rf.close()
        for wf in wfl:
            wf.close()

    def get_job_path(self):
        if self.has_job:
            return self.this_job_path
        else:
            print("A working job directory has not been created yet.")
            
    def write_metadata(self):
        # Example of script_params
        '''
        script_params = {
            'nprocs':2,
            'data_dir':'./dataset/testset',
            'debug': True    
        }
        '''                
        metadata = {}
        metadata['parameters'] = {}
        
        # default processor type
        metadata['parameters']['proc_type'] = "CPU"
                
        # additional meta data
        metadata['parameters']['nnodes'] = self.nnodes
        
        for key, value in self.script_params.items():
            metadata['parameters'][key] = value

        with open(self.this_job_path + "/meta-job.json", "w") as json_file:
            json.dump(metadata, json_file)
        
    def submit(self,job_title=""):
        # make arguments list to one string
        argstr = ' '.join(self.args)
        ##### Set Job Title #####
        if job_title != "":
            job_title = job_title.replace(" ","_") # Replace spaces with hyphens
            self.job_title = job_title
        else:
            self.job_title = ""
        ##### Make dir for new job #####
        self.make_job_path()
        ##### Make Classes File #####
        self.make_classes()
        ##### Make Job List and Batch Info #####
        self.make_job_list()
        self.make_batch_info()
        ##### request submit job (register job to database) - API Call #####
        self._request_submit_job()
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
        ##### Write Meta Data JSON File #####
        self.write_metadata()
        # Request Job Submission
        self.training = self._request_to_portal()
    
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

    def _request_submit_job(self):
        data = {
          'screenName': self.user_id,
          'title': self.job_title,
          'targetType': '83', # targetType 83 is for de job
          'workspaceName': self.workspace_name,
          'location': self.output_path # real job path
        }
        if self.debug:
            print(data)        
        response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/studio-submit-de-job', data=data)
        if response.status_code == 200:
            self.job_id = response.json()            
            print("Job ID-{} was submitted.".format(self.job_id))
        else:
            print("A problem occured when generating the job.")
        print("Job was generated in database.")    
            
    def _request_to_portal(self):
        print("Job Requested to Portal.")
        # Request to Portal to call slurm script
        self._run_slurm_script()
        # if there is no error
        return True
    
    def _run_slurm_script(self):
        data = {
          'screenName': self.user_id,
          'location': self.output_path
        }
        if self.debug:
            print(data)
        print("Running Slurm script...")
        response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/slurm-de-job-run', data=data)
        # waiting for slurm job id
        time.sleep(3)
        try:
            with open(self.this_job_path+'/job.id','r') as f:
                idstr = f.readline()
                idstr = idstr.strip()
                print("Batch job ID-{} is running on the HPC.".format(idstr))
                self.slurm_job_id = int(idstr)
                self._request_update_status("RUNNING")
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
            status = self._request_get_status_return()
            if status == "SUCCESS": # Already finished, Don't change the status
                return
            if status == "FAILED": # Already finished, Don't change the status
                return
            response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/studio-update-status', data=data)
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
            response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/get-de-job-data', data=data)
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
    
    def _request_get_status_return(self):
        try:
            data = {
              'deJobId': self.job_id
            }
            response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/get-de-job-data', data=data)
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
            response = requests.post(self.apiurl+'/api/jsonws/SDR_base-portlet.dejob/slurm-de-job-cancel', data=data)
            if response.status_code == 200:
                print("The job was successfully canceled.")
                self._request_update_status("CANCELLED")
        except:
            print("Error: Slurm Job Not Found.")