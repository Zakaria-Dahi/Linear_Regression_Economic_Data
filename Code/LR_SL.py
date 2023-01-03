import os.path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import subprocess
from LR_INT import LR_INT


class LR_SL(LR_INT):
    """
    implement linear regression using Scikitlean
    """
    def __init__(self):
        self.k = 1
        self.y_test = []
        self.y_pred = []

    def linear_regression(self, *args):
        threshold = args[0];
        city = args[1];
        serie = args[2];
        year = args[3];
        show = args[4];
        df = pd.read_csv('../Input/data.csv')  # import the CSV
        df2 = df.loc[(df['location_name'] == city) & (df['serie_name'] == serie) & (df['year'] == year)] # just change the name of the city, serie and year you want to predict
        if len(df2['value']) > threshold:
            # example by wrking on a smaller dataset. For each instruction is the same explanation as above
            df2 = df[:][:threshold]  # selec only 500 datasets
        sns.lmplot(x="period", y="value", data=df2, order=2, ci=None)  # plots the scatter plot
        #df2.fillna(method='ffill', inplace=True)  # eliminates the Nans
        # Training the model
        x = np.array(df2['period']).reshape(-1, 1)  # reshape the vector into an np array: this the feature
        y = np.array(df2['value']).reshape(-1, 1)  # reshape the vector into an array: this is the target
        #df2.dropna(inplace=True)  # removes all the rows with Null in it
        X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.25)  # its a method of sciekitlearn to split the dataset into training and testing, where 0.25 corresponds to the percentage of the datasamples to be considered as testing set
        # it splits the x and y each one appart. It will produce two training sets and two testing sets for "period" and for "value".
        regr = LinearRegression()  # create the object linearRegression of scikitlearn
        regr.fit(X_train, y_train)  # this trains the linear model, meaning making the linear model fit the data
        # print(regr.score(X_test,y_test))  # I think the higher the value, the better. NB: this return the coefficient of determination R^2 defined as (1-U/V) => full defintion of U and V available here https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        y_pred = regr.predict(X_test)  # this predicts the y prediction using the linear regression model, its the line!
        plt.scatter(X_test, y_test, color='b')  # plot the original features and the target
        plt.scatter(X_test, y_pred, color='k')  # plot the training features and the predicted target
        self.y_pred = y_pred
        self.y_test = y_test
        if show == 1:
            plt.show()  # it does not fit well so the data is not suitable for linear regression. IF so happens, one can try to reduce the data we are working on
        else:
            if os.path.exists("../Output/SL/") == False:
                os.makedirs("../Output/SL/")
            fig_name = "../Output/SL/sl_linear_prediction"+city+"_"+ serie +"_" + str(year) +".png"
            plt.savefig(fig_name)


    def display_result(self,*args):
        y_test = args[0];
        y_pred = args[1];
        city = args[2];
        serie = args[3];
        year = args[4];
        verbose = args[5];
        # the evaluation metrics for regression models: MAE and MSE: Mean Absolute Error and Mean Square Error, and Root Mean Square error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
        rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
        if verbose ==1:
            print("----------------------------------------")
            print("The Evaluation Metrics of the Linear Regression")
            print("----------------------------------------")
            print(f"The Mean Absolute Error Value is: {mae}")
            print(f"The Mean Squared Error is: {mse}")
            print(f"The Root Mean Squared Error is:{rmse}")
        else:
            txt_name = "../Output/SL/sl_linear_prediction"+city+"_"+ serie +"_" + str(year) +".txt"
            subprocess.run(["touch",txt_name]) # create the txt to store the results
            with open(txt_name,'w') as f:
                f.write("The Mean Absolute Error Value is:"+ str(mae))
                f.write('\n')
                f.write("The Mean Squared Error is:"+ str(mse))
                f.write('\n')
                f.write("The Root Mean Squared Error is:"+ str(rmse))