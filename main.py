import os 
import sklearn
import xgboost

import dill as pickle
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score, RocCurveDisplay, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from skops import hub_utils
from skops import card

# load in in the Suicide Detection dataset
# accessible at https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
df = pd.read_csv(
    "Suicide_Detection.csv",
    usecols=["text", "class"],
    dtype= {"text":str,"class":str}
)


# separate text and target class
X = df['text'].to_list()
y = df['class'].apply(lambda x: 1 if x == 'suicide' else 0).to_list()

# construct training and testing splits 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def preprocessor(s):
    """preprocessor for the tfidf vectorizer"""

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    stopwords_set = set(ENGLISH_STOP_WORDS)

    def filter(text):
        if text == None:
            return ""
        words = str(text).split()
        filtered_words = [word for word in words if word and word.lower() not in stopwords_set]
        return " ".join(filtered_words)

    return filter(s)

# construct the model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(preprocessor=preprocessor, ngram_range=(1, 3), min_df=100)),
    ('classifier', xgboost.XGBClassifier())
], verbose=True)

# fit the model using the training split
model.fit(X_train, y_train)

# use the trained model to make predictions on the testing set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# save the trained model
model_filename = "model.pkl"
with open(model_filename, mode="bw") as f:
    pickle.dump(model, file=f)

local_repo = Path("suicide-detector")

# construct the hugging face page 
hub_utils.init(
    model=model_filename,
    requirements=[f"scikit-learn={sklearn.__version__}", f"xgboost={xgboost.__version__}"],
    dst=str(local_repo),
    task="text-classification",
    data=X_test,
)

# made a header card from the metadata
model_card = card.Card(model, metadata=card.metadata_from_config(local_repo))

# add license
model_card.metadata.license = "mit"


model_description = """
Suicide Detection text classification model.

PYTHON 3.9 ONLY
"""

model_card.add(**{"Model description": model_description})

model_card.delete("Model description/Intended uses & limitations")
model_card.delete("Model Card Contact")
model_card.delete("Citation")


# model_card.delete("Evaluation Results")
model_card.delete("Model Card Authors")


training_procedure = """
Trained using 0.7 of the the Suicide and Depression Detection dataset (https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

The model vectorises each text using a trained tfidf vectorizer and then classifies using xgboost.

See main.py for further details.
"""
model_card.add(**{"Model description/Training Procedure": training_procedure})


# add description of how the model was evaluated
eval_descr = (
    "The model was evaluated on a 0.3 holdout split using f1 score, accuracy, confusion matrix and ROC curves."
)
model_card.add(**{"Model Evaluation": eval_descr})

# compute model evaluation metrics and add details to the hugging face model card
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="micro")

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
disp.figure_.savefig(local_repo / "confusion_matrix.png")

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
disp.plot()
disp.figure_.savefig(local_repo / "roc_curve.png")


clf_report = classification_report(
    y_test, y_pred, output_dict=True, target_names=["not suicide", "suicide"]
)

model_card.add_metrics(**{"accuracy": accuracy, "f1 score": f1,"ROC AUC": roc_auc})
model_card.add_plot(**{"Model Evaluation/Confusion matrix": "confusion_matrix.png"})
model_card.add_plot(**{"Model Evaluation/ROC Curve": "roc_curve.png"})


clf_report = pd.DataFrame(clf_report).T.reset_index()
model_card.add_table(
    **{
        "Classification Report": clf_report,
    },
)


get_started_code = """
```python
import sklearn 
import dill as pickle

from skops import hub_utils
from pathlib import Path

suicide_detector_repo = Path("./suicide-detector")

hub_utils.download(
    repo_id="AndyJamesTurner/suicideDetector",
    dst=suicide_detector_repo
)

with open(suicide_detector_repo / "model.pkl", 'rb') as file:
    clf = pickle.load(file)

classification = clf.predict(["I want to kill myself"])[0]
```
"""

authors = """
This model was created by the following authors:

* Andy Turner
"""

# add additional details to the page including 
# model description, getting started guide, and author
model_card.add(**{
    "How to Get Started with the Model": get_started_code,
    "Model Authors": authors
    }
)


# construct a readme from the model card
model_card.save(local_repo / "README.md")

# add this file to the repo to document how it was constructed
hub_utils.add_files(
    os.path.realpath(__file__),
    dst=str(local_repo),
)