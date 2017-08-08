
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

error_messages = {
    "No clear target in training data": 
        ("The training data must have " 
         "exactly one more column than " 
         "the test data."),
    "Training data has too many columns":
        ("The training data has more "
         "than one column different than "
         "the testing data: %s"),
    "Column names inconsistent":
        ("The training columns and the "
         "test columns must have "
         "identical names excepts for "
         "the target variables. "
         "Different columns: %s")
    }

def X_y_split(X_train, X_test):
    """
    Determines which variables are the target
    and which are the features. Returns just
    The X and y data in the training dataset
    as a tuple.
    
    Example usage:
    X, y = learn.X_y_split(X_train, X_test)
    
    Parameters
    ----------
    X_train: pandas dataframe
        The data that has the target in it.
    
    X_test: pandas dataframe
        The data that does not have the target in it.
    """
    X_train = X_train.copy()
    n_train_cols = X_train.shape[1]
    n_test_cols = X_test.shape[1]
    
    if n_train_cols != n_test_cols + 1:
        msg = error_messages["No clear target in training data"]
        raise ValueError(msg)
        
    test_columns = set(X_test.columns)
    train_columns = set(X_train.columns)
    target_columns = train_columns - test_columns
    if len(target_columns) > 1:
        key = "Training data has too many columns"
        msg_ = error_messages[key]
        msg = msg_ % str(target_columns)
        raise ValueError(msg)

    extra_columns_in_test = test_columns - train_columns
    if extra_columns_in_test:
        key = "Column names inconsistent"
        msg_ = error_messages[key]
        msg = msg_ % str(extra_columns_in_test)
        raise ValueError(msg)     

    y_name = target_columns.pop()
    y = X_train.pop(y_name)
    return X_train, y


def make_data(source, 
              missing_data=None, 
              categorical=None, 
              outliers=None):
    """
    Utility function to assist in loading different 
    sample datasets. Returns training data (that 
    contains the target) and testing data (that
    does not contain the target).
    
    Parameters
    ----------
    source: string, optional (default="boston")
        The specific dataset to load. Options:
        - Regression: "boston", "diabetes"
        - Classification: "cancer", "digits", "iris"
        
    missing_data: bool or NoneType (default=None)
        To be implemented
        Determines if there is missing data
        
    categorical: bool or NoneType (default=None)
        To be implemented
        Determines if there is categorical data
        
    outliers: bool or NoneType (default=None)
        To be implemented
        Determines if there are outliers in the dataset
    """
    if source == "boston":
        data = datasets.load_boston()
    elif source == "diabetes":
        data = datasets.load_diabetes()
        data["feature_names"] = ["f{}".format(v) 
                                 for v in range(10)]
    elif source == "cancer":
        data = datasets.load_breast_cancer()
    elif source == "digits":
        data = datasets.load_digits()
        data["feature_names"] = ["f{}".format(v) 
                                 for v in range(64)]        
    elif source == "iris":
        data = datasets.load_iris()
    X = pd.DataFrame(data=data.data, 
                     columns=data.feature_names)
    y = pd.Series(data=data.target)
    X_train, X_test, y_train, _ = train_test_split(X, 
                                                   y, 
                                                   test_size=.5,
                                                   random_state=42)
    X_train["target"] = y_train
    return X_train, X_test


def is_classification_problem(y, max_classes="auto"):
    """
    Check if a target variable is a classification
    problem or a regression problem. Returns True if
    classification and False if regression. On failure,
    raises a ValueError.
    
    Parameters
    ----------
    y: array-like
        This should be the target variable. Ideally, 
        you should convert it to be numeric before 
        using this function.
        
    max_classes: int or float, optional (default="auto")
        Determines the max number of unique target values
        there can be for classification problems
        
        If "auto" - sets it equal to 10% of the dataset or
            100, whichever is smaller
        If float - interprets as percent of dataset size
        If int - interprets as number of classes
    """
    y = pd.Series(y)
    n = len(y)
    n_unique = len(y.unique())
    if max_classes == "auto":
        n_max_classes = int(n*.1)
        max_classes = min(n_max_classes, 100)
    if isinstance(max_classes, float):
        n_max_classes = int(n*max_classes)
        max_classes = min(n_max_classes, int(n/2))
    # If y is numeric
    if y.dtype.kind in 'bifc':
        # If there are more than max_classes
        # classify as a regression problem
        if n_unique > max_classes:
            return False
        # If there are floating point numbers
        # classify as a regression problem
        decimals = (y - y.astype(int)).mean()
        if decimals > .01:
            return False
    if n_unique <= max_classes:
        return True
    try:
        y.astype(float)
        return False
    except ValueError:
        msg = ("Malformed target data. "
               "Target is non-numeric "
               "and there are more "
               "unique values than allowed "
               "by max_classes")
        raise ValueError(msg)