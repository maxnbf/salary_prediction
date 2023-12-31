import math 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
import pandas as pd
import matplotlib.pyplot as plt

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
    

def calc_salary(row):
    """
        Calculate the salary for a row in the pandas dataframe given the data used

        Parameters:
            row - row from the original job_postings.csv
        Return:
            the expected annual salary of the job corresponding to this row
    """
    dct = {
        "YEARLY": 1,
        "MONTHLY": 12,
        "HOURLY": 2000
    }

    mx = row["max_salary"]
    mn = row["min_salary"]
    md = row["med_salary"]

    if not math.isnan(md):
        return dct[row["pay_period"]] * md
    if not math.isnan(mx) and not math.isnan(mn):
        return dct[row["pay_period"]] * (mx + mn)/2

    return None



def remove_stopwords_numbers_punctionation_and_lemmatize(desc):
    """
        Removes stopwords, numbers, and punctuation from the text description

        More: Lemmatize aqnd Stem words

        Parameters:
            desc - Original job description corresponding to one row
        Return:
            a cleaned text description
    """
    #remove numbers, and links
    desc = re.sub('[0-9]+\S+|\s\d+\s|\w+[0-9]+|\w+[\*]+.*|\s[\*]+\s|www\.[^\s]+','', desc)
    words = word_tokenize(desc)
    filtered = list(filter(lambda w: w.lower() not in stop_words, words))
    filtered = list(filter(lambda w: w not in punctuation, filtered))


    # lemmatizer = WordNetLemmatizer()
    # stemmer = PorterStemmer()
    # filtered = [lemmatizer.lemmatize(word) for word in filtered]
    # filtered = [stemmer.stem(word) for word in filtered]
    return ' '.join(filtered)



def tokenize(desc):
    """
        Tokenize a job description, and remove any tokens of length 1 

        Parameters: 
            desc - cleaned job description
        Return:
            tokenized job description
    """
    tokens = nltk.word_tokenize(desc)

    tokens = [token for token in tokens if len(token) > 1]
    return tokens


# for RNN it was recommended we use lengths of 50 tokens.
def lengths(desc):
    """
        Calculate the length of a job description. This is used to help us study the data we are working with

        Parameters
            desc - tokenized job description
        Return:
            Number of tokens
    """
    return len(desc)


def pre_process_descriptions(descriptions):
    """_summary_

    Args:
        descriptions (list-like): iterable of cleaned string descriptions
    """
    res = []
    for desc in descriptions:
        res.append(remove_stopwords_numbers_punctionation_and_lemmatize(desc))
    return res


def percent_of_predictions_in_range(actual, pred, range):
    """
        Calculates the percentage of predictions that were within a specific range of their actual targets

        Parameters:
            actual - list of actual salaries
            pred - list of predicted salaries
            range - value we are checking for delta in actual and pre

        Return
            percent of guesses in range
    """
    count = 0
    for a, p in zip(actual, pred):
        if abs(a-p) <= range:
            count += 1
    
    return (count / len(pred)) * 100

def predict_and_analyze(model, data, actuals): 
    """
        Uses the given model to predict the results of the given data, and report metrics

        Parameters:
            model - specific model to test
            data - featurized data that will be input into the model
            actuals - target salary data

        Return
            preds - predictions of given data
            mae - mean absolute error
            mse - mean squared error
    """
    preds = model.predict(data)

    mse = mean_squared_error(actuals, preds)
    print(f"Mean Squared Error (MSE): {mse}")

    mae = mean_absolute_error(actuals, preds)
    print(f"Mean Absolute Error: {mae}")

    print("Percent of postings predicted within $5,000: ", "{:.2f}".format(percent_of_predictions_in_range(actuals, preds, 5000)))
    print("Percent of postings predicted within $10,000: ", "{:.2f}".format(percent_of_predictions_in_range(actuals, preds, 10000)))

    return preds, mae, mse

def get_evaluation_metric(eval_func, all_pred_y, true_y, func_args=None):
    """
    Takes a list of predictions, the true values, and an evaluation function. Gets the performance
    metric returned from the evaluation function for each prediction that was made.
    Args:
        eval_func (callable): evaluation metric to be called on the predictions
        all_pred_y (list of list): all y predictions 
        true_y (list): true_y values
        func_args (any, optional): For evaluation functions that require an argument (max 1), pass it here. Defaults to None.

    Returns:
        list: the metric requested for each prediction
    """
    metrics = []
    for preds in all_pred_y:
        
        if func_args:
            metric = eval_func(true_y, preds, func_args)
        else:
            metric = eval_func(true_y, preds)

        metrics.append(metric)
    return metrics

def plot_accuracy_in_buckets(actual, preds):
    """
        Creates $5000 buckets for comparing the predicted salaries to their actual salaries, and plots the results

        Parameters:
            actual - list of test target values
            preds - list of predicted target values
    """
    absolute_differences = [abs(actual - predicted) for actual, predicted in zip(actual, preds)]

    # Combine the lists into a DataFrame
    data = pd.DataFrame({'Predictions': preds, 'Actual Salaries': actual, 'Absolute Difference': absolute_differences})

    # Define absolute difference buckets at intervals of 5000 up to 40000
    difference_buckets = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, float('inf')]

    # Create a new column in the DataFrame to store the bucket for each absolute difference
    data['Absolute Difference Bucket'] = pd.cut(data['Absolute Difference'], bins=difference_buckets, labels=['<5000', '<10000', '<15000', '<20000', '<25000', '<30000', '<35000', '<40000', '<45000', '<50000', ' >=50000'], right=False)

    # Group the data by the absolute difference buckets and count the number of predictions in each bucket
    bucket_counts = data.groupby('Absolute Difference Bucket').size().reset_index(name='Count')

    # Plot the data
    plt.bar(bucket_counts['Absolute Difference Bucket'], bucket_counts['Count'])
    plt.xlabel('Absolute Difference Bucket')
    plt.ylabel('Number of Predictions')
    plt.title('Distribution of Predictions by Absolute Difference Bucket')
    plt.xticks(fontsize=7)
    plt.show()