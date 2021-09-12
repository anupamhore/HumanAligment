import pandas as pd
from app_logging.logger import appLogger
from data_preprocessing.preprocessing import Preprocessor
import os
from sys import platform
from feature_selection.featureSelection import FeatureSelection
from sklearn.model_selection import train_test_split
from best_model_finder.tuner import Model_Finder
from file_ops.file_methods import File_Operation

class TrainModel:
    def __init__(self):
        self.logger = appLogger()
        self.file_object = open("Training_Logs/TrainingLog.txt", 'a+')
        pass

    def modelTraining(self):
        """
        Method Name: modelTraining
        Description:This class trains the dataset after doing all the preprocessing
        Output: None
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object,"Start of Training!!!")
        separator = "\\"
        if platform == 'darwin':  # mac system
            separator = "/"

        try:
            #get the data
            self.df = pd.read_csv('Training_Data/Input.csv')

            # initiate the preprocessor class
            preprocessor = Preprocessor(self.file_object, self.logger)

            # map the target variables name to the numerical numbers(since model will expect numbers)
            preprocessor.map_target_variable(self.df,'Label')

            self.logger.log(self.file_object, "Mapped Variable: %s"%self.df['Label'].value_counts())

            # separate the dependant and independant variables

            X, Y = preprocessor.separate_label_features(self.df, 'Label')

            # find out if 0 are present in the dataset. Change them to their mean()
            X = preprocessor.imputeZeros(X)


            #find if any missing value present in the data
            isnullpresent, missingValueColumns = preprocessor.is_null_present(X)


            #impute missing values
            if isnullpresent:
                X = preprocessor.impute_missingValues(X,missingValueColumns)

            # initial columns to drop
            cols_to_drop = ['Unnamed: 0', '# Columns: time']
            X = preprocessor.dropVariables(X, cols_to_drop, 1, True)

            #outlier treatment
            cols_needing_outlierTreatment = ['avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23','var_rss23']
            distributionTypes =['Skewed','Highly Skewed','Gaussian','Highly Skewed','Gaussian','Skewed']

            isLengthSame, X = preprocessor.outlier_treatment(X, cols_needing_outlierTreatment,distributionTypes)

            if isLengthSame:

                # Separate the categorical and continous variables from the input variables for processing
                cagtegoricalVars, numericalVars = preprocessor.separate_cat_num(X)

                self.logger.log(self.file_object, "Categorical Variables: {}".format(cagtegoricalVars))
                self.logger.log(self.file_object, "Numerical Variables: {}".format(numericalVars))


                # check for multi-collinearity
                high_collinear_vars = preprocessor.checkforMultiCollinearity(X, 0.7)
                self.logger.log(self.file_object, "High Multi-Collinear Variables: {}".format(high_collinear_vars))

                # perform feature selection
                featureSelection = FeatureSelection(self.logger)

                # find features whose VIF values are less than 10. Those features will be good for the model
                features_with_low_vif = featureSelection.findVIF_Factor(X)
                self.logger.log(self.file_object, "Low VIF Variables: {}".format(features_with_low_vif))

                # find constant features
                const_features = featureSelection.findConstantFeatures(X)
                # we will drop the constant features
                if len(const_features) > 0:
                    X.drop(const_features, axis=1, inplace=True)

                best_features = featureSelection.findbestFeatures(X,Y)
                self.logger.log(self.file_object, "Best Features: {}".format(best_features))
                """
                Feature Selection will be performed based on the following variables
                     # 1. high_collinear_vars
                     # 2. features_with_low_vif
                     # 3. best_features
                """

                # find out final features for model
                X_features = featureSelection.findFinalFeatures(high_collinear_vars,best_features, features_with_low_vif)

                self.logger.log(self.file_object, "Final Features before transformation: {}".format(X_features))

                # replace the final features in the dataset from the feature selection process
                X = featureSelection.updateDataSet(X, X_features)

                self.logger.log(self.file_object, "Final Features after transformation: {}".format(X.columns))

                # split the data
                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30,
                                                                    random_state=100)

                model_finder = Model_Finder(self.logger)

                # getting the best model for each of the clusters
                best_model_name, best_model = model_finder.get_best_model(X_train, y_train, X_test, y_test)

                # save the best model to the directory
                file_op = File_Operation(self.logger)
                save_model = file_op.save_model(best_model, best_model_name)


            self.logger.log(self.file_object, "Training Successfull!!!")

        except Exception as e:
            self.logger.log(self.file_object, "Training Failure: %s"%Exception(e))
            raise Exception(e)





