import json
import os

class SessionCreator:
    def __init__(self, name):
        self.results_path = os.path.join(name, 'results')
        self.logs_path = os.path.join(name, 'logs')
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)

    @property
    def get_results_path(self):
        return self.results_path

    @property
    def get_logs_path(self):
        return self.logs_path

