import csv
import os
from LR_INT import LR_INT;
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tf.disable_v2_behavior()

class LR_TF(LR_INT):
    """
    implement linear regression using Tensorflow V1
    """

    def __init__(self):
        self.log = [];
        self.train_x = [];
        self.train_y= [];
        self.test_x= [];
        self.test_y= [];



    def linear_regression(self, *args):
        """
        implement linear regression using Tensorflow
        """
        city = args[1];
        serie = args[2];
        year = args[3];
        show = args[4];
        learning_rate = args[5]  # defines the learning rate
        training_epochs = args[6]  # the training epochs
        display_step = args[7]  # the display step

        # asarray is a function that converts the data to arrays, the data could be lists, dicts, etc.
        df = pd.read_csv('../Input/data.csv')  # import the CSV
        df2 = df.loc[(df['location_name'] == city) & (df['serie_name'] == serie) & (df['year'] == year)] # just change the name of the city, serie and year you want to predict
        # prepare the training data sample 70% of the whole dataset, and the remaining 30% are made for testing purposes
        X_data = np.asarray(np.array(df2['period']).tolist());
        Y_data = np.asarray(np.array(df2['value']).tolist());
        train_x = X_data[:round(0.7*len(X_data))]
        train_y = Y_data[:round(0.7*len(X_data))]
        test_x = X_data[round(0.7*len(X_data)):]
        test_y =  Y_data[round(0.7*len(Y_data)):]
        n_samples = train_x.shape[0]  # recover the number of training features
        # this defines input entrance of input data
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32)
        # set the model weights and bias of our linear equation wx + b
        W = tf.Variable(np.random.rand(), name="weight")
        B = tf.Variable(np.random.rand(), name="bias")
        # build up the linear model
        linear_model = W * X + B
        # Set the loss as the one half mean squared error
        cost = tf.reduce_sum(tf.square(linear_model - Y) / (2 * n_samples))  # 1/2.M \sum | y^ - y| => This formula is the same as the one of Andrew: one half Mean Square Error
        # gradient descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # intiialise the vartiables
        init = tf.global_variables_initializer()

        with tf.Session() as sess:  # session with S in  uppercase
            # loadintiailised variables in current session
            sess.run(init)
            # Fit all the training data
            for epoch in range(training_epochs):
                # gradient decent
                sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
                # display information
                if (epoch + 1) % display_step == 0:
                    c = sess.run(cost, feed_dict={X: train_x, Y: train_y})
                    if show == 1:
                        print("Epoch:{0:6} \t Cost:{1:10.4} \t W:{2:6.4} \t b:{3:6.4}".format(epoch + 1, c, sess.run(W),sess.run(B)))
                    else:
                        self.log.append(["epoch:",epoch + 1, "cost function:" , c, "W:", sess.run(W),"B:",sess.run(B)])

            # train the linear model
            self.training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
            # test the linear model
            self.testing_cost = sess.run(tf.reduce_sum(tf.square(linear_model - Y)) / (2 * test_x.shape[0]), feed_dict={X: test_x, Y: test_y})  # this is the mean square error: the same as the Coursera formula
            # recover the variables
            self.B = sess.run(B);
            self.W = sess.run(W);
            self.train_x = train_x;
            self.train_y = train_y;
            self.test_x = test_x;
            self.test_y = test_y;


    def display_result(self,*args):
        log = args[0];
        training_cost = args[1];
        testing_cost = args[2];
        city = args[3];
        serie = args[4];
        year = args[5];
        W = args[6];
        B = args[7];
        train_x = args[8];
        train_y = args[9];
        test_x = args[10];
        test_y = args[11];
        varbose = args[12];
        """
        Display results of linear regression using Tensorflow
        """
        # Display the numerical reslts of the training and testing
        if varbose == 1:
            print("Final training cost:", training_cost, "W:", W, "B:", B, '\n')
            print("Final testing cost:", testing_cost)
            print("Absolute mean square loss difference:", abs(training_cost - testing_cost))
        else:
            if os.path.exists("../Output/TF/") == False:
                os.mkdir("../Output/TF/")
            with open("../Output/TF/tf_linear_prediction" + city + "_" + serie + "_" + str(year) + ".csv",
                      'w') as f:
                writer = csv.writer(f)
                writer.writerows(log)

            if os.path.exists("../Output/TF/") == False:
                os.makedirs("../Output/TF/")
            with open("../Output/TF/tf_linear_prediction" + city + "_" + serie + "_" + str(year) + ".txt",
                      'w') as f:
                f.write("Final testing cost: " + str(testing_cost))
                f.write("\n")
                f.write("Absolute mean square loss difference: " + str(abs(training_cost - testing_cost)))
        # Display the fitted line one training data
        plt.plot(train_x, train_y, 'ro', label="Original data")
        plt.plot(train_x, W * train_x + B,label="Linear Model")  # makes sense we extract the nex prediceted data by applyin the WX+B model formula
        plt.legend()
        if varbose == 1:
            plt.show()
        else:
            if os.path.exists("../Output/TF/") == False:
                os.mkdir("../Output/TF/")
            fig_name = "../Output/TF/tf_linear_prediction_training_data" + city + "_" + serie + "_" + str(year) + ".png"
            plt.savefig(fig_name)
            plt.close()

        # Display the fitted line one test data
        plt.plot(test_x, test_y, "bo", label="Testing data")
        plt.plot(train_x, W * train_x + B, label="Linear Model")
        plt.legend()
        if varbose == 1:
            plt.show()
        else:
            if os.path.exists("../Output/TF/") == False:
                os.mkdir("../Output/TF/")
            fig_name = "../Output/TF/tf_linear_prediction_testing_data" + city + "_" + serie + "_" + str(year) + ".png"
            plt.savefig(fig_name)