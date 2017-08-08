
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
from learn import utils

class Regression():
    def __init__(self, time_to_compute=None):
        self.time_to_compute = time_to_compute
        
    def fit(self, X, y):
        model = RandomForestRegressor(n_estimators=100, 
                                      oob_score=True)
        model.fit(X, y)
        self.model = model
        self.oob_predictions = model.oob_prediction_
        self.score_type = "R2"
        self.score = r2_score(y, self.oob_predictions)
        return self
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
    
    
class Classification():
    def __init__(self, time_to_compute=None):
        """
        """
        self.time_to_compute = time_to_compute
        
    def fit(self, X, y):
        """
        Currently y must be numeric. Wrap 
        LabelVectorizer as TODO.
        """
        y = pd.Series(y)
        self.n_classes = len(y.unique())
        model = RandomForestClassifier(n_estimators=100, 
                                       oob_score=True)
        model.fit(X, y)
        self.model = model
        
        # Evaluation metrics
        if self.n_classes == 2:
            self.oob_predictions = model.oob_decision_function_[:, 1]
            self.score_type = "AUC"
            self.score = roc_auc_score(y, self.oob_predictions)
        else:
            self.oob_predictions = model.oob_decision_function_
            self.score_type = "AUC"
            y_bin = label_binarize(y, sorted(pd.Series(y).unique()))
            self.score = roc_auc_score(y_bin, 
                                       self.oob_predictions)
        return self
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    
class All():
    def __init__(self, time_to_compute=None):
        self.time_to_compute = time_to_compute
        
    def fit(self, X, y):
        self.classification = utils.is_classification_problem(y)
        if self.classification:
            model = Classification()
        else:
            model = Regression()
        model.fit(X, y)
        self.model = model
        self.score = model.score
        self.score_type = model.score_type
        return self
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions