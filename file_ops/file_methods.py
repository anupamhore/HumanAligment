import pickle
import os
import shutil

class File_Operation:

    def __init__(self,logger):
        self.logger = logger
        self.file_object = open("Training_Logs/ModelFileOps.txt", 'a+')
        self.model_directory = 'models/'

    def save_model(self, model, filename):
        """
          Method Name: save_model
          Description:This class saves the respective model with the filename in the model directory
          Output: None
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object,'save_model Started!!!')
        try:
            path = os.path.join(self.model_directory, filename)

            if os.path.isdir(path):
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)

            with open(path + "/" + filename + '.sav', 'wb') as f:
                pickle.dump(model, f)

            self.logger.log(self.file_object, 'save_model Completed!!!')
            return 'success'

        except Exception as e:
            self.logger.log(self.file_object, 'save_model Error: %s'%Exception(e))
            raise Exception(e)

    def load_model(self, filename):
        """
          Method Name: load_model
          Description:This class load the model for testing
          Output: The ML Model
          Written By: Anupam Hore
          Version: 1.0
          Revisions: None
        """
        self.logger.log(self.file_object, 'load_model Started!!!')
        try:
            with open(self.model_directory + filename + "/" + filename + '.sav', 'rb') as f:

                self.logger.log(self.file_object, 'load_model Completed!!!')
                return pickle.load(f)

        except Exception as e:
            self.logger.log(self.file_object, 'load_model Error: %s' % Exception(e))
            raise Exception(e)


