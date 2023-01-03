from LR_SL import LR_SL
from LR_TF import LR_TF


def main():
    """
    Help:
    Common paramters:
        city: choose one of the 52 cities available (e.g. Ceuta)
        serie: choose one of the 33 seconomical metrics (e.g. companies with X employees, etc).
        year: choose a year from 2003 to 2007
        verbose: set to "1" if you want visual display of the results and "0" if you want to store the results in files
        Frame: set to "1" for scikit learn and "0" for tensorflow linear regression
    Tensorflow parameters:
        lr: the learning rate of the gradient decent.
        training epochs: for how many epochs the training is performed.
        display_rate: the laps of iterations at which we recover the log of the execution
    """
    # -_-_-_-_-_-_- Parameters _-_-_-_-_-_-_-_-_-_-_

    frame = 0;
    city  = "Ceuta"
    serie = "Men Activity Percentage"
    year = 2003
    verbose = 0
    lr = 0.01 # the learning rate of the gradient descent
    train_epochs = 2000 # training epochs
    display_rate = 200 # onl

    # _-_-_-_-_-_ The call to the univariate linear regression _-_-_-_-__-_-
    test_frame = lambda frame: frame == 1;
    var = LR_SL() if test_frame(frame) == True else LR_TF();
    var.linear_regression(500,city,serie,year,verbose,lr,train_epochs,display_rate)

    if  test_frame(frame):
       var.display_result(var.y_test, var.y_pred, city, serie, year, verbose)
    else:
       var.display_result(var.log,var.training_cost,var.testing_cost,city,serie,year,var.W,var.B,var.train_x,var.train_y, var.test_x, var.test_y, verbose)

if __name__ == "__main__":
    main()


