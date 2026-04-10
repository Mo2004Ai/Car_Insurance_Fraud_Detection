import preprosses as cifd
import numpy as np

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# DATA
X = cifd.final_df.drop('fraud_reported', axis=1)
y = cifd.final_df['fraud_reported']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# BALANCING
ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)
rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
smote = SMOTE(random_state=42)

X_train, y_train = ros.fit_resample(X_train, y_train)
X_train, y_train = rus.fit_resample(X_train, y_train)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 1 - Decision Tree (MAIN)
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("===== Decision Tree =====")
print(classification_report(y_test, dt_pred))

# 2 - XGBoost (MAIN)
xgb_model = XGBClassifier(
    colsample_bytree=1.0,
    learning_rate=0.2,
    max_depth=7,
    n_estimators=200,
    subsample=0.7,
    scale_pos_weight=7,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("===== XGBoost =====")
print(classification_report(y_test, xgb_pred))

# 3 - Voting
voting_model = VotingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
        ('xgb', XGBClassifier(max_depth=5, learning_rate=0.1, eval_metric='logloss'))
    ],
    voting='soft'
)

voting_model.fit(X_train, y_train)
voting_pred = voting_model.predict(X_test)

print("===== Voting =====")
print(classification_report(y_test, voting_pred))

# 4 - Stacking
stacking_model = StackingClassifier(
    estimators=[
        ('dt', DecisionTreeClassifier(max_depth=3, random_state=42)),
        ('xgb', XGBClassifier(max_depth=5, learning_rate=0.1, eval_metric='logloss'))
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=True
)

stacking_model.fit(X_train, y_train)
stack_pred = stacking_model.predict(X_test)

print("===== Stacking =====")
print(classification_report(y_test, stack_pred))

# THRESHOLD FUNCTION
def find_best_threshold(model, X_test, y_test, name):
    probs = model.predict_proba(X_test)[:, 1]

    best_t, best_f1 = 0, 0

    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_test, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    final_preds = (probs >= best_t).astype(int)

    print(f"\n===== {name} =====")
    print("Best Threshold:", best_t)
    print("Accuracy:", accuracy_score(y_test, final_preds))
    print("Precision:", precision_score(y_test, final_preds))
    print("Recall:", recall_score(y_test, final_preds))
    print("F1:", f1_score(y_test, final_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, final_preds))

    return best_t

dt_t = find_best_threshold(dt_model, X_test, y_test, "Decision Tree")
xgb_t = find_best_threshold(xgb_model, X_test, y_test, "XGBoost")
voting_t = find_best_threshold(voting_model, X_test, y_test, "Voting")
stack_t = find_best_threshold(stacking_model, X_test, y_test, "Stacking")