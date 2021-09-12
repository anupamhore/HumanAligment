import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import mutual_info_classif

class FeatureSelection:
    """
        This class shall  be used to select the best possible features for model training

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
    """
    def __init__(self, logger):
        self.logger = logger
        self.file_object = open("Training_Logs/FeatureSelection.txt", 'a+')


    def findVIF_Factor(self, X):
        """
        Method Name: findVIF_Factor
        Description: This function finds out the variables with low VIF factors. Those variables
                     can be considered for model training

        Parameter: X ( independant variables)
        Output: list of variables with low VIF
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object,"findVIF_Factor method Started!!!!")
        try:
            scaler = StandardScaler()
            arr = scaler.fit_transform(X)
            vif = pd.DataFrame()
            vif['VIF'] = [variance_inflation_factor(arr, i) for i in range(arr.shape[1])]
            vif['Feature'] = X.columns
            series = vif[vif['VIF'] < 10]
            arr = list(series.Feature)

            self.logger.log(self.file_object, "findVIF_Factor method Completed!!!!")

            return arr

        except Exception as e:
            self.logger.log(self.file_object, "findVIF_Factor error: %s"%Exception(e))
            raise Exception(e)

    def findFinalFeatures(self,high_collinear_vars, best_features, features_with_low_vif):
        """
        Method Name: findFinalFeatures
        Description: This function filters out the variables from the given arguments and gives the best
                     features for model training

        Parameter: high_collinear_vars, best_features,  features_with_low_vif
        Output: list of best variables for model training
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object,"findFinalFeatures Started!!!")

        try:

            #First combine the low performing variables
            combined_list_bad = high_collinear_vars

            combined_good_list = features_with_low_vif + best_features
            combined_good_list_set = set(combined_good_list)
            combined_good_list = list(combined_good_list_set)

           #Second remove them from the high performing variables in features_with_low_vif
            final_features = [feature for feature in combined_good_list if feature not in combined_list_bad]

            self.logger.log(self.file_object, "findFinalFeatures Completed!!!")
            #self.file_object.close()
            return final_features

        except Exception as e:
            self.logger.log(self.file_object, "findFinalFeatures Failed error: %s"%Exception(e))
            #self.file_object.close()
            raise Exception(e)

    def updateDataSet(self,X,X_features):
        """
        Method Name: updateDataSet
        Description: This function updates the dataset with the features selected after feature selection process

        Parameter: X(dataset), X_features(features important for the model)
        Output: updated dataset
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "updateDataSet Started!!!")
        try:
            drop_list = [feature for feature in X.columns if feature not in X_features]
            X.drop(drop_list,axis=1,inplace=True)

            self.logger.log(self.file_object, "updateDataSet Completed!!!")
           # self.file_object.close()
            return X

        except Exception as e:
            self.logger.log(self.file_object, "Update Dataset Failed: %s"%Exception(e))
           # self.file_object.close()
            raise Exception(e)

    def findConstantFeatures(self,data):
        """
        Method Name: findConstantFeatures
        Description: This function helps to find out the variables with constant values.
                     Sometime, the data might be constant values. These variables are not
                     good for model traning

        Parameter: X(dataset), X_features(features important for the model)
        Output: list of const columns
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "findConstantFeatures Started!!!")
        try:
            var_threshold = VarianceThreshold(threshold=0)
            var_threshold.fit(data)
            #var_threshold.get_support() returns True,False for each variable
            const_columns = [column for column in data.columns if column not in data.columns[var_threshold.get_support()]]
            self.logger.log(self.file_object, "findConstantFeatures Completed!!!")

            return const_columns

        except Exception as e:
            self.logger.log(self.file_object, "findConstantFeatures Fail: %s"%Exception(e))

            raise Exception(e)

    def findbestFeatures(self,X,Y):
        """
        Method Name: findbestFeatures
        Description: This function helps to find out the 5 best features from the lot

        Parameter: X(independant vars), Y(target vars)
        Output: list of best feature columns
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object,"findbestFeatures Started!!!")
        try:
            mutual_info = mutual_info_classif(X, Y)
            mutual_info = pd.Series(mutual_info)
            mutual_info.index = X.columns
            mutual_info.sort_values(ascending=False)

            df = mutual_info.to_frame()
            df = df.reset_index()
            arr = df.to_numpy()
            sorted_best_features = []
            count = 0
            for list_ in arr:
                if count < 5:
                    sorted_best_features.append(list_[0])
                count = count + 1
            self.logger.log(self.file_object, "findbestFeatures Completed!!!")
            return sorted_best_features

        except Exception as e:
            self.logger.log(self.file_object, "findbestFeatures Fail: %s"%Exception(e))
            raise Exception(e)


