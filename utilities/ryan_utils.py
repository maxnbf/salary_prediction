from sklearn.svm import SVR
import utilities.data_utils as dutil
import sklearn.metrics as smetrics
import matplotlib.pyplot as plt

def train_and_predict_SVR(kernel, X_train, y_train, X_test, c_values):
    """_summary_

    Args:
        kernel (string): The kernel function for SVR
        X_train (list of list): X_training data
        y_train (list): List of training_y data
        X_test (list of list): Matrix of input data to test
        c_values (list): list of c values to test for each SVR model

    Returns:
        _type_: a list of models that were trained and a list of predictions that each model made on the x_test data
    """
    EPSILON_CONSTANT = 1000

    all_preds, all_models = [], []
    for c in c_values:  
        model = SVR(kernel=kernel, C=c, epsilon=EPSILON_CONSTANT)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        all_preds.append(preds)
        all_models.append(model)
    return all_models, all_preds


def graph_SVR_performance_by_C(title, all_preds, y_actual, C_values):
    """
    Graphs performance of SVR models by the quality of the predictions they made on the test data.
    'performance' is the percent of predictions they made within range of the actual data.

    Args:
        title (string): Title of the graph
        all_preds (list of list): list of all predictions made by each model
        y_actual (list): actual y values
        C_values (list): list of c values used in the models
    """
    within_5000 = dutil.get_evaluation_metric(dutil.percent_of_predictions_in_range, all_preds, y_actual, func_args=5000)
    within_10000 = dutil.get_evaluation_metric(dutil.percent_of_predictions_in_range, all_preds, y_actual, func_args=10000)
    within_15000 = dutil.get_evaluation_metric(dutil.percent_of_predictions_in_range, all_preds, y_actual, func_args=15000)
    
    plt.xlabel("SVR c-parameter value")
    plt.ylabel("Pct predictions within x-dollars")
    plt.plot(C_values, within_5000,  "-bo", label="x=$5000")
    plt.plot(C_values, within_10000, "--gd", label="x=$10,000")
    plt.plot(C_values, within_15000, ":y+", label="x=$15,000")
    plt.legend()
    plt.title(title)

def graph_performance_by_error(title, all_preds, y_actual, C_values):
    """
    Graphs performance of SVR models by the quality of the predictions they made on the test data.
    'performance' is the amount of error each model had between its predictions and the true y_values.

    Args:
        title (string): Title of the graph
        all_preds (list of list): list of all predictions made by each model
        y_actual (list): actual y values
        C_values (list): list of c values used in the models
    """
    mae = dutil.get_evaluation_metric(smetrics.mean_absolute_error, all_preds, y_actual)
    mse = dutil.get_evaluation_metric(smetrics.mean_squared_error, all_preds, y_actual)
    
    plt.xlabel("SVR C-parameter Value")
    plt.ylabel("Mean Average Error in Dollars")
    plt.plot(C_values, mae,  "-bo")
    plt.title(title)