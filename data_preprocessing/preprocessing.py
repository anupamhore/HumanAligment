import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before training.

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None

    """
    def __init__(self, file_object,logger):
        self.logger = logger
        self.file_object = file_object

    def map_target_variable(self,df,labelName):
        """
        Method Name: map_target_variable
        Description: This function maps the target variable names to numerical numbers
        Parameter: df,labelName ( Dataframe and the target Variable name)
        Output: None
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "map_target_variable method Started!!!")
        try:

            df[labelName] = df[labelName].map({'bending1': 1, 'bending2': 2, 'cycling': 3, 'lying': 4, 'sitting': 5, 'standing': 6, 'walking': 7})

            self.logger.log(self.file_object, "map_target_variable method Completed!!!")

        except Exception as e:
            self.logger.log(self.file_object, "Mapping Error: %s"%Exception(e))
            raise Exception(e)

    def separate_label_features(self, df, labelName):
        """
        Method Name: separate_label_features
        Description: This function separates the independant and dependant variables
        Parameter: df,labelName ( Dataframe and the target Variable name)
        Output: dependant variables, independant variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        try:
            self.logger.log(self.file_object, "separate_label_features method Started")
            X = df.drop(labelName, axis=1)
            Y = df[labelName]
            self.logger.log(self.file_object, "separate_label_features method Completed")
            return X, Y
        except Exception as e:
            self.logger.log(self.file_object, "Error in separating the target and independant variables %s"%Exception(e))
            raise Exception(e)

    def imputeZeros(self,df):
        """
        Method Name: imputeZeros
        Description: This function finds out the 0 in the data set and change them to their mean()
        Parameter: df(Dataframe)
        Output: The dataframe with zero imputed
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "imputeZeros method Started!!!")
        try:
            for feature in df.columns:
                df[feature] = df[feature].replace(0, df[feature].mean())

            self.logger.log(self.file_object, "imputeZeros method Completed!!!")
            return df

        except Exception as e:
            self.logger.log(self.file_object, "imputeZeros Error: %s"%Exception(e))
            raise Exception(e)


    def is_null_present(self, df):
        """
        Method Name: is_null_present
        Description: This function validates if any column in the dataframe has any missing values
        Parameter: df(the dataframe)
        Output: True or False( if missing values present in the dataset) & the columns which have missing values
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "Missing Value Search Started")
        null_present = False
        cols = df.columns
        columns_with_missing_values = []
        try:

            null_counts = df.isnull().sum()
            for i in range(len(null_counts)):
                if null_counts[i] > 0:
                    null_present = True
                    columns_with_missing_values.append(cols[i])

            if null_present:
                self.logger.log(self.file_object,"Missing Value Columns are %s"%columns_with_missing_values)
            self.logger.log(self.file_object,"Missing Values Search Completed")
            return null_present, columns_with_missing_values

        except Exception as e:
            self.logger.log(self.file_object,"Missing Value Search Error: %s" %Exception(e))
            raise Exception(e)

    def impute_missingValues(self, df, cols):
        """
        Method Name: impute_missingValues
        Description: This function imputes the missing values based on the type.
                     For continous numerical variable we use median()
                     For discrete numerical variable we use mode()
                     For categorical variable we use mode()
        Parameter: df(the dataframe), cosl ( the missing value columns list)
        Output: Dataset with missing values imputed
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "impute_missingValues Started!!!")
        try:
            for col in cols:
                if df[col].dtypes == 'object':
                    df[col] = df[col].fillna(df[col].mode())
                else:
                    discreteCol = pd.Categorical(df[col])
                    if len(discreteCol.categories) > 10:
                        #continous data
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        # discrete variable
                        df[col] = df[col].fillna(df[col].mode())

            self.logger.log(self.file_object, "impute_missingValues Completed!!!")
            return df

        except Exception as e:
            self.logger.log(self.file_object, "Error imputing missing values %s" %Exception(e))
            raise Exception(e)

    def impute_infs(self,df,cols):
        """
        Method Name: impute_infs
        Description: This function imputes the infs values present in the dataset
        Parameter: df(the dataframe), cosl ( the missing value(infs) columns list)
        Output: Dataset with missing values imputed
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "impute_infs Started!!!")
        try:
            for col in cols:
                if np.isinf(df[col]).sum() > 0:
                    x = df[col]
                    x[np.isneginf(x)] = x.median()

            self.logger.log(self.file_object, "impute_infs Completed!!!")
            return df

        except Exception as e:
            self.logger.log(self.file_object, "Infs imputation error: %s"%Exception(e))
            raise Exception(e)


    def dropVariables(self,df,cols,axis,inplace):
        """
        Method Name: dropVariables
        Description: This function drops the respective cols from the dataframe
        Parameter: df(the dataframe), cols ( the columns), axis (0,1), inplace(True,False)
        Output: Dataset with dropped variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None

        """
        self.logger.log(self.file_object, "dropVariables Started!!!")
        try:
            df.drop(cols,axis=axis,inplace=inplace)
            self.logger.log(self.file_object, "dropVariables Completed!!!")
            return df

        except Exception as e:
            self.logger.log(self.file_object, "Error in dropping variables due to %s"%Exception(e))
            raise Exception(e)


    def outlier_treatment(self,df, cols,distType):
        """
        Method Name: outlier_treatment
        Description: This function is responsible for the outlier treatment
        Parameter: df(the dataframe), cols ( the columns), distType(list of variable distribution types)
                   Depending on the distribution, we will treat the outliers for the variables
        Output: Dataset with outlier imputed
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
        """
        self.logger.log(self.file_object, "outlier_treatment Started!!!")
        isLengthEqual = True
        try:
            if len(cols) == len(distType):
                count = 0
                for col in cols:
                    distributionType = distType[count]

                    if distributionType == "Gaussian":
                        upper_bound = df[col].mean() + 3 * df[col].std()
                        lower_bound = df[col].mean() - 3 * df[col].std()
                        df[col].clip(upper=upper_bound, inplace=True)
                        df[col].clip(lower=lower_bound, inplace=True)

                    elif distributionType == "Skewed":
                        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
                        upper_bound = df[col].quantile(0.75) + 1.5 * IQR
                        lower_bound = df[col].quantile(0.25) - 1.5 * IQR
                        df[col].clip(upper=upper_bound, inplace=True)
                        df[col].clip(lower=lower_bound, inplace=True)

                    else:
                        IQR = df[col].quantile(0.75) - df[col].quantile(0.25)
                        upper_bound = df[col].quantile(0.75) + 3 * IQR
                        lower_bound = df[col].quantile(0.25) - 3 * IQR
                        df[col].clip(upper=upper_bound, inplace=True)
                        df[col].clip(lower=lower_bound, inplace=True)

                    count = count + 1
                self.logger.log(self.file_object, "outlier_treatment Completed!!!")
                return isLengthEqual, df

            else:
                isLengthEqual = False
                self.logger.log(self.file_object, "Length of columns and length of distributions mismatch!!!")
                return isLengthEqual, df

        except Exception as e:
            self.logger.log(self.file_object, "Outlier Imputation Error: %s"%Exception(e))
            raise Exception(e)

    def separate_cat_num(self,df):
        """
        Method Name: separate_cat_num
        Description: This function separates the categorical variables and the numerical variable
        Parameter: df(the dataframe)
        Output: categorical vars, numerical vars
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None:
        """
        self.logger.log(self.file_object, "separate_cat_num Started!!!")
        self.logger.log(self.file_object,"X columns: %s"%df.columns)
        try:
            categorical_features = []
            numerical_features = []
            features = df.columns
            for feature in features:
                if df[feature].dtypes == 'object':
                    categorical_features.append(feature)
                else:
                    #check if the feature is a discrete variable or not
                    discreteVar = pd.Categorical(df[feature])
                    if len(discreteVar.categories) > 19:
                        #that means its numerical
                        numerical_features.append(feature)
                    else:
                        categorical_features.append(feature)

            self.logger.log(self.file_object, "separate_cat_num Completed!!!")
            return categorical_features, numerical_features

        except Exception as e:
            self.logger.log(self.file_object, "Cannot Separate Categorical and Numerical Variables: %s"%Exception(e))
            raise Exception(e)



    def checkforMultiCollinearity(self, df, threshold):
        """
        Method Name: checkforMultiCollinearity
        Description: This function find out the variables which have high multi-collinearity among themselves.

        Parameter: df(the dataframe), threshold(cut off value till what multi-collinearity is accepted)
        Output: List of high multi-collinear variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None:
        """
        self.logger.log(self.file_object, "checkforMultiCollinearity Started!!!")
        try:
            corr_set = set()
            corr_matrix = df.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > threshold:
                        colName = corr_matrix.columns[i]
                        corr_set.add(colName)

            self.logger.log(self.file_object, "checkforMultiCollinearity Completed!!!")
            return list(corr_set)

        except Exception as e:
            self.logger.log(self.file_object, "Finding Multi-Collinearity Failed: %s"%Exception(e))
            raise Exception(e)

    def scaleData(self, data):
        """
        Method Name: scaleData
        Description: This function scales the dataset in same unit using StandardScaler

        Parameter: data(the dataframe)
        Output: List of scaled variables
        On Failure: Exception

        Written By: Anupam Hore
        Version: 1.0
        Revisions: None:
        """
        self.logger.log(self.file_object, "scaleData method Started!!!")
        self.data = data
        try:
            scaler = StandardScaler()
            arr = scaler.fit_transform(self.data)
            self.logger.log(self.file_object, "scaleData method Completed!!!")
            return arr
        except Exception as e:
            self.logger.log(self.file_object, "StandardScaler Conversion Failed: %s" % Exception(e))
            raise Exception(e)









