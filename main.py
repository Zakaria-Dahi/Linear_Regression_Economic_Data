from LR import LR

def main():
    """
    city: choose one of the 52 cities
    serie: choose one of the economical metrics e.g. companies with X employees, etc.
    year: choose a year from 2003 to 207
    verbose: set to "1" if you want visual display of the results and "0" if you want to store the results in files

    """
    var = LR();
    city  = "Ceuta"
    serie = "Men Activity Percentage"
    year = 2003
    verbose = 0
    var.linear_regression(500,city,serie,year,verbose) # 1: verbose, others: Not verbose
    var.display_result(var.y_test,var.y_pred,city,serie,year,verbose)


if __name__ == "__main__":
    main()
