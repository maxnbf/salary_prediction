from sklearn.svm import SVR
import utilities.data_utils as dutil
import sklearn.metrics as smetrics
import matplotlib.pyplot as plt

def train_and_predict_SVR(kernel, X_train, y_train, X_test, y_test):
    EPSILON_CONSTANT = 1000
    C_VALUES = [1, 10, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000]
    all_preds = []
    for c in C_VALUES:  
        model = SVR(kernel=kernel, C=c, epsilon=EPSILON_CONSTANT)
        model.fit(X_train, y_train)
        preds = dutil.predict_and_analyze(model, X_test, y_test)
        all_preds.append(preds)
    return all_preds


def graph_SVR_performance_by_C(title, all_preds, y_actual, C_values):
    mae = dutil.get_evaluation_metric(smetrics.mean_absolute_error, all_preds, y_actual)
    mse = dutil.get_evaluation_metric(smetrics.mean_squared_error, all_preds, y_actual)
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