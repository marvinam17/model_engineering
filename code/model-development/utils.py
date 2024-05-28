from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def evaluate_model(y_true, y_pred, y_prob):
    """
    Diese Funktion nimmt einen Vektor von realen Daten, einen Vektor von Vorhersagen und einen Verktor von Wahrscheinlichkeiten und 
    berechnen Genauigkeit, F1, Precision, Recall und AUC.
    Die Ergebnisse werden in einem Dictionary zurückgegeben und zusätzlich in die Konsole ausgegeben.
    """
    metrics_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc_score" : roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    for item in metrics_dict.keys():
        print(item, ":  ", metrics_dict[item])
        
    return metrics_dict

def create_sampling_datasets(samplers, X, y):
    """
    Erstellt Sampling Datensätze
    Input Args:
    samplers: Dict von Samplerklassen
    X: Features
    y: Klassen
    Return Args:
    data: {"sampler":(Features, Klassen)}
    """
    data = {}
    for i_sampler in samplers.keys():
        df_resampled, y_resampled = samplers[i_sampler].fit_resample(X, y)
        data[i_sampler] = (df_resampled, y_resampled)
    return data

def create_train_test_sampled_split(df_all_features, df_selected_features):
    """
    Erstellt einen Trainings und Testsplit für zwei DataFrames
    Input Args:
    df_all_features: DataFrame
    df_selected_features: DataFrame
    Return Args:
    prep_data: {"Featureset":{{"X_train": X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}}}
    """
    prep_data = {}
    X_all_features = df_all_features.copy()
    y_all_features = X_all_features.pop("success")
    X_selected_features = df_selected_features.copy()
    y_selected_features = X_selected_features.pop("success")

    (X_train, 
     X_test, 
     y_train, 
     y_test) = train_test_split(X_all_features, 
                                y_all_features, 
                                test_size=0.2, 
                                random_state=42)
    prep_data["All Features"] = {"X_train": X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test}

    (X_train_feature_selection,
     X_test_feature_selection,
     y_train_feature_selection, 
     y_test_feature_selection) = train_test_split(X_selected_features, 
                                                  y_selected_features, 
                                                  test_size=0.2, 
                                                  random_state=42)
    prep_data["Selected Features"] = {"X_train": X_train_feature_selection, 
                                  "X_test":X_test_feature_selection, 
                                  "y_train":y_train_feature_selection, 
                                  "y_test":y_test_feature_selection}
    return prep_data

def evaluate_model_metrics(y_true, y_pred, y_prob)
    """ 
    Diese Funktion nimmt einen Vektor von realen Daten, einen Vektor von Vorhersagen und einen Verktor von Wahrscheinlichkeiten und 
    berechnen Genauigkeit, F1, Precision, Recall und AUC.
    Die Ergebnisse werden in einem Dictionary zurückgegeben.
    """
    metrics_dict = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auc_score" : roc_auc_score(y_true, y_prob)
    } 
    return metrics_dict

def train_and_evaluate(clf, X_train, y_train, X_test, y_test):
    """
    Trainiert und evaluiert ein ML Klassifizierer.
    Input Args:
    clf: sklearn Klassifizierer
    X_train: Trainingsdaten
    y_train: Trainingslabels
    X_test: Testdaten
    y_test: Testlabels
    Output Args:
    metrics_dict aus evaluate_model_metrics 
    """
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    return evaluate_model_metrics(y_test, y_pred, y_prob)