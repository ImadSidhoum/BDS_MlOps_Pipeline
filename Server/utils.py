
# import the required libraries
from __future__ import print_function
import pickle
import os.path
import io,sys
import shutil
import requests
from mimetypes import MimeTypes
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

import zipfile
import os.path
import glob

def zipdirectory(filezip, pathzip):
    lenpathparent = len(pathzip)+1   ## utile si on veut stocker les chemins relatifs
    def _zipdirectory(zfile, path):
        for i in glob.glob(path+'\\*'):
            if os.path.isdir(i): _zipdirectory(zfile, i )
            else:

                zfile.write(i, i[lenpathparent:]) ## zfile.write(i) pour stocker les chemins complets
    zfile = zipfile.ZipFile(filezip,'w',compression=zipfile.ZIP_DEFLATED)
    _zipdirectory(zfile, pathzip)
    zfile.close()

def dezip(filezip, pathdst = ''):
    if pathdst == '': pathdst = os.getcwd()  ## on dezippe dans le repertoire locale
    zfile = zipfile.ZipFile(filezip, 'r')
    for i in zfile.namelist():  ## On parcourt l'ensemble des fichiers de l'archive

        if os.path.isdir(i):   ## S'il s'agit d'un repertoire, on se contente de creer le dossier
            try: os.makedirs(pathdst + os.sep + i)
            except: pass
        else:
            try: os.makedirs(pathdst + os.sep + os.path.dirname(i))
            except: pass
            data = zfile.read(i)                   ## lecture du fichier compresse
            fp = open(pathdst + os.sep + i, "wb")  ## creation en local du nouveau fichier
            fp.write(data)                         ## ajout des donnees du fichier compresse dans le fichier local
            fp.close()
    zfile.close()


class DriveAPI:
    global SCOPES
      
    # Define the scopes
    SCOPES = ['https://www.googleapis.com/auth/drive']
  
    def __init__(self):
        
        # Variable self.creds will
        # store the user access token.
        # If no valid token found
        # we will create one.
        self.creds = None
  
        # The file token.pickle stores the
        # user's access and refresh tokens. It is
        # created automatically when the authorization
        # flow completes for the first time.
  
        # Check if file token.pickle exists
        if os.path.exists('token.pickle'):
  
            # Read the token from the file and
            # store it in the variable self.creds
            with open('token.pickle', 'rb') as token:
                self.creds = pickle.load(token)
  
        # If no valid credentials are available,
        # request the user to log in.
        if not self.creds or not self.creds.valid:
  
            # If token is expired, it will be refreshed,
            # else, we will request a new one.
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                self.creds = flow.run_local_server(port=0)
  
            # Save the access token in token.pickle
            # file for future usage
            with open('token.pickle', 'wb') as token:
                pickle.dump(self.creds, token)
  
        # Connect to the API service
        self.service = build('drive', 'v3', credentials=self.creds)
  
        # request a list of first N files or
        # folders with name and id from the API.
        results = self.service.files().list(
            pageSize=100, fields="files(id, name)").execute()
        items = results.get('files', [])
  
        # print a list of files
  
        #print("Here's a list of files: \n")
        #print(*items, sep="\n", end="\n\n")
        #print(items, type(items))
  
    def FileDownload(self, file_name):
        results = self.service.files().list(
            pageSize=100, fields="files(id, name)").execute()
        items = results.get('files', [])
        file_id = None
        for item in items:
            if item["name"] == file_name:
                file_id = item["id"]
                break

        if file_id is None:
            print("Something went wrong, file not found.")
            return False

        results = self.service.files().list(
            pageSize=100, fields="files(id, name)").execute()
        items = results.get('files', [])
        print(items, type(items))

        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
          
        # Initialise a downloader object to download the file
        downloader = MediaIoBaseDownload(fh, request, chunksize=204800)
        done = False
  
        try:
            # Download the data in chunks
            while not done:
                status, done = downloader.next_chunk()
  
            fh.seek(0)
              
            # Write the received data to the file
            with open(file_name, 'wb') as f:
                shutil.copyfileobj(fh, f)
  
            print("File Downloaded")
            # Return True if file Downloaded successfully
            return True
        except:
            
            # Return False if something went wrong
            print("Something went wrong.")
            return False
  
    def FileUpload(self, filepath):
        
        # Extract the file name out of the file path
        name = filepath.split('/')[-1]
          
        # Find the MimeType of the file
        mimetype = MimeTypes().guess_type(name)[0]
          
        # create file metadata
        file_metadata = {'name': name}
  
        try:
            media = MediaFileUpload(filepath, mimetype=mimetype)
              
            # Create a new file in the Drive storage
            file = self.service.files().create(
                body=file_metadata, media_body=media, fields='id').execute()
              
            print("File Uploaded.")
          
        except:
              
            # Raise UploadError if file is not uploaded.
            raise UploadError("Can't Upload File.")
  
def compare(new_run_id):
    json_response = requests.get(f'http://127.0.0.1:5001/version')
    old_run_id = json_response.text 
    new_run_info = mlflow.get_run(run_id=new_run_id)
    try: 
        lod_run_info = mlflow.get_run(run_id=old_run_id)

        new_acc = new_run_info.data.metrics['training_score']
        old_acc = old_run_info.data.metrics['training_score']

        if (new_acc>old_acc):
            f_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            zipdirectory(f_name+'.zip', './../src_image/mlruns/0/'+new_run_id+'/artifacts/model')
            FileUpload(f_name+'.zip')
            data = json.dumps({"signature_name": "serving_default", "name": f_name+'.zip',"version":str(new_run_id)})
            headers = {"content-type": "application/json"}
            json_response = requests.post(f'http://127.0.0.1:5001/update', data=data, headers=headers)
            print(json_response)
    except:
        print('something went wrong')