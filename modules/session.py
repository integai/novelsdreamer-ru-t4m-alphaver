import os
import yaml

class SessionCreator:
    def __init__(self, session_name: str, prepare: bool):
        self.session_name = self.prepare_session(session_name) if prepare else session_name
        self.create_session_folders()
        self.create_metadata()
    
    def prepare_session(self, session_name):
        path_name = 'sessions'
        os.makedirs(path_name, exist_ok=True)
        return os.path.join(path_name, session_name)
    
    def create_session_folders(self):
        # Main folder
        os.makedirs(self.session_name, exist_ok=True)
        # Sub folders
        self.logs_path = self.create_subfolder('logs')
        self.results_path = self.create_subfolder('results')
        self.support_files_path = self.create_subfolder('support')
    
    def create_subfolder(self, subfolder_name):
        path = os.path.join(self.session_name, subfolder_name)
        os.makedirs(path, exist_ok=True)
        return path
    
    def save_data_to_yaml(self, data):
        with open(data, 'w') as f:
            yaml.dump(self.metadata, f)
    
    def create_metadata(self):
        # Creating metadata for use in training
        name = f'{self.session_name}/metadata.yaml'
        if not os.path.isfile(name):
            metadata = {
                'session_paths': {
                    'session_name': self.session_name,
                    'logs_path': self.logs_path,
                    'results_path': self.results_path,
                    'support_files_path': self.support_files_path
                }
            }
            with open(name, 'w') as f:
                yaml.dump(metadata, f)
    
    def get_metadata(self):
        name = f'{self.session_name}/metadata_session.yaml'
        with open(name, 'r') as f:
            metadata = yaml.load(f, Loader=yaml.FullLoader)
        return metadata
