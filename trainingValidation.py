import pandas as pd
import os
from os import listdir
from glob import glob
from app_logging.logger import appLogger
from sys import platform
import csv
import shutil

class train_validation:
    def __init__(self):
        self.logger = appLogger()
        self.file_path = open("Validation_Logs/ValidationLog.txt", 'a+')


    def startValidation(self):
        """
           Method Name: startValidation
           Description: This method will read all the csv files from the respective directory and
                        clean them, read them into dataframes, combine them and save the combined
                        file again to a master csv file which will be used for model training
           Output: None
           On Failure: Raise ValueError,KeyError,Exception

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """
        self.logger.log(self.file_path,"startValidation Started!!!")
        try:
            files = []
            start_dir = os.getcwd() + "\Raw_Data"
            pattern = "*.csv"

            for dir, _, _ in os.walk(start_dir):
                files.extend(glob(os.path.join(dir, pattern)))

            self.logger.log(self.file_path,"Total Files are:%s"%len(files))
            self.logger.log(self.file_path, "Files are:%s" %files)

            # delete the extra entries in the csv files
            self.cleanFiles(files)

            # combine the files for each directory and put the labels
            self.combileFiles()


            self.logger.log(self.file_path,"startValidation Completed!!!")

        except Exception as e:
            self.logger.log(self.file_path,"Validation Error: %s"%Exception(e))
            raise Exception(e)

    def cleanFiles(self,files):
        """
           Method Name: cleanFiles
           Description: This method will clean the unwanted entries from the file
                        and will save the file with the clear entries which we can
                        read into a dataframe
           Output: None
           On Failure: Raise ValueError,KeyError,Exception

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """
        self.logger.log(self.file_path, "cleanFiles Started!!!")
        start_dir =  os.getcwd() +"\Clean_Raw_Data"
        try:
            for file in files:
                with open(file, 'r') as f:
                    lines = f.readlines() # Read the files

                separator = "\\"
                if platform == 'darwin': # mac system
                    separator = "/"
                filePathSplit = file.split(separator)
                filename = filePathSplit.pop()
                filedirectory = filePathSplit.pop()

                directoryPath = start_dir + separator + filedirectory
                if not os.path.isdir(directoryPath):
                    os.makedirs(directoryPath)

                file_to_write_path = directoryPath + separator + filename
                self.logger.log(self.file_path, "Path Name: %s"%file_to_write_path)
                with open(file_to_write_path, 'w') as f: # Remove the 4 rows which are common in all the files
                    count = 1
                    for line in lines:
                        if count > 4:
                            f.write(line) # re-write the entries in the existing file only after 4th row
                        count = count + 1

                #check if entries in the files are valid. Sometime, single row may also be present,
                # instead of multiple columns. In that case we need to handle the file properly
                firstLine = ''
                file_needs_correction = False
                with open(file_to_write_path,"r") as f:
                    next(f)
                    data = csv.reader(f)
                    for line in enumerate(data):
                        if len(line[1]) == 1:
                            file_needs_correction = True
                            break
                if file_needs_correction:
                    """
                    Three steps are performed
                    Step 1: We will collect the first row which contains the header of the dataset
                    """
                    with open(file_to_write_path, "r") as f:
                        lines = f.readlines()
                        count = 0
                        for line in lines:
                            if count == 0:
                                firstLine = line
                                break

                    """
                    Three steps are performed
                    Step 2: We will collect all the items in the file except the first line which is the header
                            Header we already collected in the first step
                    """
                    with open(file_to_write_path, "r") as f:
                        next(f)
                        lines1 = f.readlines()

                    """
                    Three steps are performed
                    Step 3: We will write into the same file (newly) with the header printed first
                            and then the other items
                            The other items are present in the form of string. So, we will first split
                            them and then join the items with ",". 
                            NOTE: Splitting added one "\n" character so we have to delete it and hence the
                            pop operation.    
                    """
                    with open(file_to_write_path, "w+") as f:
                        f.write(firstLine)
                        for line in lines1:
                            list_ = line.split(" ")
                            list_.pop()
                            item = ','.join(list_)
                            f.write(item)
                            f.write('\n')


            self.logger.log(self.file_path, "cleanFiles Completed!!!")

        except Exception as e:
            self.logger.log(self.file_path, "File cleaning Error: %s"%Exception(e))
            raise Exception(e)

    def combileFiles(self):
        """
           Method Name: combileFiles
           Description: This method will combile all the files of each directory
                        and will assign the labels(target variable)
           Output: None
           On Failure: Raise ValueError,KeyError,Exception

           Written By: Anupam Hore
           Version: 1.0
           Revisions: None
        """
        self.logger.log(self.file_path, "combileFiles Started!!!")
        try:
            separator = "\\"
            if platform == 'darwin':  # mac system
                separator = "/"
            start_dir = os.getcwd() + "\Clean_Raw_Data"
            files = [f for f in listdir(start_dir)]

            #create an empty master dataframe which will have all the combined csv files from all
            #the directories and then finally we will convert it to master csv file and that file
            #will be used for training

            master_df = pd.DataFrame() # Empty dataset
            for directory in files:
                files_ = [f for f in listdir(start_dir + separator + directory)]
                self.logger.log(self.file_path, "Total Files: %s" % len(files_))

                df = pd.DataFrame() #empty dataframe for sub directories
                for file in files_:

                    csv_path = start_dir + separator + directory + separator + file
                    self.logger.log(self.file_path, "File Path: %s" % csv_path)
                    tmp_df = pd.read_csv(csv_path, error_bad_lines=False)
                    df = pd.concat([df, tmp_df], ignore_index=True)
                df['Label'] = str(directory)
                master_df = pd.concat([master_df, df], ignore_index=True)
            self.logger.log(self.file_path, "Master DataFrame Created!!!")

            training_csv_dir = os.getcwd() + "\Training_Data"
            if not os.path.isdir(training_csv_dir):
                os.makedirs(training_csv_dir)

            finalfileName = 'Input.csv'
            finalPath = training_csv_dir + separator + finalfileName
            master_df.to_csv(finalPath)

            self.logger.log(self.file_path, "combileFiles Completed!!!")

        except Exception as e:
            self.logger.log(self.file_path, "File Combine Error: %s"%Exception(e))
            raise Exception(e)










