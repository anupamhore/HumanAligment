from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_auc_score

class Model_Finder:

   def __init__(self,logger):
      self.logger = logger
      self.file_object = open("Training_Logs/BestModelLog.txt", 'a+')



   def get_best_params_for_logisticRegression_lbfgs(self,X_train, y_train):

      """
        Method Name: get_best_params_for_logisticRegression1
        Description:This class trains the data based on hyper parameter tuning
        Input: X_train, y_train
        Output: model name, model itself
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
      """
      self.logger.log(self.file_object, "get_best_params_for_logisticRegression_lbfgs Started!!!")
      try:
         log_reg = LogisticRegression(solver='lbfgs', multi_class='ovr')
         log_reg.fit(X_train, y_train)
         self.logger.log(self.file_object, "get_best_params_for_logisticRegression_lbfgs Completed!!!")
         return log_reg


      except Exception as e:
         self.logger.log(self.file_object, "logisticRegression_lbfgs Error: %s"%Exception(e))
         raise Exception(e)

   def get_best_params_for_logisticRegression_newton_cg(self, X_train, y_train):

      """
        Method Name: get_best_params_for_logisticRegression1
        Description:This class trains the data based on hyper parameter tuning
        Input: X_train, y_train
        Output: model name, model itself
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
      """
      self.logger.log(self.file_object, "get_best_params_for_logisticRegression_newton_cg Started!!!")
      try:
         log_reg = LogisticRegression(solver='newton-cg', multi_class='ovr')
         log_reg.fit(X_train, y_train)
         self.logger.log(self.file_object, "get_best_params_for_logisticRegression_newton_cg Completed!!!")
         return log_reg


      except Exception as e:
         self.logger.log(self.file_object, "logisticRegression_newton_cg Error: %s" % Exception(e))
         raise Exception(e)

   def get_best_params_for_logisticRegression_saga(self, X_train, y_train):

      """
        Method Name: get_best_params_for_logisticRegression1
        Description:This class trains the data based on hyper parameter tuning
        Input: X_train, y_train
        Output: model name, model itself
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
      """
      self.logger.log(self.file_object, "get_best_params_for_logisticRegression_saga Started!!!")
      try:
         log_reg = LogisticRegression(solver='saga', multi_class='ovr')
         log_reg.fit(X_train, y_train)
         self.logger.log(self.file_object, "get_best_params_for_logisticRegression_saga Completed!!!")
         return log_reg


      except Exception as e:
         self.logger.log(self.file_object, "logisticRegression_saga Error: %s" % Exception(e))
         raise Exception(e)

   def get_best_params_for_logisticRegression_sag(self, X_train, y_train):

      """
        Method Name: get_best_params_for_logisticRegression1
        Description:This class trains the data based on hyper parameter tuning
        Input: X_train, y_train
        Output: model name, model itself
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
      """
      self.logger.log(self.file_object, "get_best_params_for_logisticRegression_saga Started!!!")
      try:
         log_reg = LogisticRegression(solver='sag', multi_class='ovr')
         log_reg.fit(X_train, y_train)
         self.logger.log(self.file_object, "get_best_params_for_logisticRegression_sag Completed!!!")
         return log_reg


      except Exception as e:
         self.logger.log(self.file_object, "logisticRegression_sag Error: %s" % Exception(e))
         raise Exception(e)

   def getModelScore(self,model,X,Y):

      """
        Method Name: getModelScore
        Description:This class calculate the model score based on y_pred, y_test
        Input: X,Y
        Output: score of the model
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
      """
      self.logger.log(self.file_object, "getModelScore Started!!!")
      try:
         y_pred1 = model.predict(X)
         labels = [1, 2, 3, 4, 5, 6, 7]
         y_pred1 = label_binarize(y_pred1, classes=labels)
         score = roc_auc_score(Y, y_pred1, average='macro', multi_class='ovr')
         self.logger.log(self.file_object, "getModelScore Completed!!!")
         return score

      except Exception as e:
         self.logger.log(self.file_object, "getModelScore Error: %s"%Exception(e))
         raise Exception(e)



   def get_best_model(self,X_train, y_train, X_test, y_test):
      """
        Method Name: get_best_model
        Description:This class trains various model with the training data set and
                    tries to find out the best model while calculating the accuracy
                    with the test data set
        Input: X_train, y_train, X_test, y_test
        Output: model name, model itself
        Written By: Anupam Hore
        Version: 1.0
        Revisions: None
      """
      self.logger.log(self.file_object,"get_best_model Started!!!")
      try:

         # Logistic Regression for solver lbfgs
         self.log_reg1 = self.get_best_params_for_logisticRegression_lbfgs(X_train, y_train)
         self.log_reg_score1 = self.getModelScore(self.log_reg1,X_test,y_test)

         # Logistic Regression for solver newton_cg
         self.log_reg2 = self.get_best_params_for_logisticRegression_newton_cg(X_train, y_train)
         self.log_reg_score2 = self.getModelScore(self.log_reg2,X_test,y_test)

         # Logistic Regression for solver saga
         self.log_reg3 = self.get_best_params_for_logisticRegression_saga(X_train, y_train)
         self.log_reg_score3 = self.getModelScore(self.log_reg3,X_test,y_test)


         # Logistic Regression for solver sag
         self.log_reg4 = self.get_best_params_for_logisticRegression_sag(X_train, y_train)
         self.log_reg_score4 = self.getModelScore(self.log_reg4,X_test,y_test)

         self.scoreList = [
            {"modelName": "Logistic Regression lbfgs", "modelscore": self.log_reg_score1, "model": self.log_reg1},
            {"modelName": "Logistic Regression newton_cg", "modelscore": self.log_reg_score2, "model": self.log_reg2},
            {"modelName": "Logistic Regression saga", "modelscore": self.log_reg_score3, "model": self.log_reg3},
            {"modelName": "Logistic Regression sag", "modelscore": self.log_reg_score4, "model": self.log_reg4}]
         self.scoreList.sort(key=lambda x: x['modelscore'], reverse=True)
         modelObject = self.scoreList[0]

         self.logger.log(self.file_object, "get_best_model Completed!!!")
         return modelObject['modelName'], modelObject['model']


      except Exception as e:
         self.logger.log(self.file_object, "get_best_model Error: %s"%Exception(e))
         raise Exception(e)



