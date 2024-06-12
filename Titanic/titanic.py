import pandas as pd
import numpy as np

path = '/content/gdrive/MyDrive/Kaggle/Titanic'

train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

####################################################################################################
####################################################################################################
####################################################################################################

def preprocessing(train, test):
    data_list = [train, test]
    for data in data_list:
        sex_dummies = pd.get_dummies(data["Sex"])

        # One-hot encode the 'Sex' column
        data['Sex'] = sex_dummies['female'].astype(int)

        # Fill missing values in the 'Age' column
        sex_mean = data.groupby("Sex")["Age"].mean()

        data.loc[(data["Sex"] == 0) & (data["Age"].isnull()), 'Age'] = sex_mean[0]
        data.loc[(data["Sex"] == 1) & (data["Age"].isnull()), 'Age'] = sex_mean[1]

        # Categorize the 'Age' column
        bins = [0,16,26,36,62,100]
        labels=[0,1,2,3,4]

        data['Age'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

        # Calculate 'FamilySize' by adding 'SibSp' and 'Parch'
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data.drop(["SibSp", "Parch"], axis=1)

        # Replace missing values in the 'Embarked' column
        most_embarked = data["Embarked"].value_counts().idxmax()
        data["Embarked"].fillna(most_embarked, inplace=True)

        # Categorize the 'Embarked' column
        em_mapping = {'S':0, 'C':1, 'Q':2}
        data['Embarked'] = data['Embarked'].map(em_mapping)

        # Replace missing values in the 'Fare' column
        data["Fare"].fillna(data["Fare"].mean(), inplace=True)

        # Categorize the 'Fare' column
        categories = pd.cut(data['Fare'], bins=4)
        bins = [interval.right for interval in sorted(categories.unique())]

        data.loc[data['Fare'] <= bins[0], 'Fare'] = 0
        data.loc[(data['Fare'] > bins[0]) & (data['Fare'] <= bins[1]), 'Fare'] = 1
        data.loc[(data['Fare'] > bins[1]) & (data['Fare'] <= bins[2]), 'Fare'] = 2
        data.loc[data['Fare'] > bins[2], 'Fare'] = 3

        # Drop unnecessary columns
        if data is train:
            data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
        else:
            data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

    return train, test

train, test = preprocessing(train, test)
test_wo_pid = test.drop(["PassengerId"], axis=1)

'''
train.isnull().sum()
test.istull().sum()
train.describe()
test.describe()
'''

####################################################################################################
####################################################################################################
####################################################################################################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X = train.drop(["Survived"], axis=1)
X['Age'] = X['Age'].astype(int)
y = train["Survived"]

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.3 , random_state = 123)

def get_max_coef_feature(coefs, features):
    max_abs_coef = 0
    max_coef_feature = None

    for coef, feature in zip(coefs, features):
        abs_coef = abs(coef)
        if abs_coef > max_abs_coef:
            max_abs_coef = abs_coef
            max_coef_feature = (coef, feature)
    
    return max_coef_feature

def select_model(X_train , X_test , y_train , y_test):

    models = {
            "Logistic Regression" : LogisticRegression(),
            "K-Nearest Neighbor" : KNeighborsClassifier(), 
            "Naive Bayes" : GaussianNB(),
            "SVC" : SVC(),
            "Decision Tree Classifier" : DecisionTreeClassifier(),
            "Random Forest Classifier" : RandomForestClassifier(),
            "Bagging Classifier" : BaggingClassifier(base_estimator=DecisionTreeClassifier()),
            "Gradient Boosting Classifier" : GradientBoostingClassifier(),
            "Ada Boost Classifier" : AdaBoostClassifier(),
            "eXtreme Gradient Boosting Classifier" : XGBClassifier()
    }

    accuracy_table = pd.DataFrame(columns=['Accuracy Score'])
    feature_importance_table = pd.DataFrame(columns=['Feature', 'Importance'])

    for model_name, model_trainer in models.items():
        model = model_trainer.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        accuracy_table.loc[model_name] = accuracy

        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_

            if feature_importance is not None:
                 max_coef_feature = get_max_coef_feature(feature_importance, X.columns)

                 feature_importance_table.loc[model_name, 'Feature'] = max_coef_feature[1]
                 feature_importance_table.loc[model_name, 'Importance'] = max_coef_feature[0]

        elif hasattr(model, 'coef_'):
            feature_importance = model.coef_

            if feature_importance is not None:
                max_coef_feature = get_max_coef_feature(model.coef_[0], X.columns)
                
                feature_importance_table.loc[model_name, 'Feature'] = max_coef_feature[1]
                feature_importance_table.loc[model_name, 'Importance'] = max_coef_feature[0]

        else:
            feature_importance = None
            feature_importance_table.loc[model_name, 'Feature'] = None
            feature_importance_table.loc[model_name, 'Importance'] = None
        
    return accuracy_table, feature_importance_table

accuracy_table, feature_importance_table = select_model(X_train , X_test , y_train , y_test)

'''
accuracy_table.sort_values(by='Accuracy Score', ascending=False)
feature_importance_table
'''

####################################################################################################
####################################################################################################
####################################################################################################

from sklearn.model_selection import GridSearchCV

gboost = GradientBoostingClassifier(random_state=123)
gboost.fit(X_train, y_train)

# Hyperparameter tuning
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(estimator=gboost, param_grid=param_grid, cv=5, 
                           scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_param = grid_search.best_params_

parma_df = pd.DataFrame({"learning_rate" :[best_param['learning_rate']], 
                         "max_depth" : [best_param['max_depth']],
                         "n_estimators" : [best_param['n_estimators']]}, index=["Parameter values"]).transpose()

# Train the model
best_model = GradientBoostingClassifier(learning_rate=best_param['learning_rate'],
                                        max_depth=best_param['max_depth'],
                                        n_estimators=best_param['n_estimators'],
                                        random_state=123)

best_model.fit(X_train, y_train)

pred = best_model.predict(test_wo_pid)


####################################################################################################
####################################################################################################
####################################################################################################

# Check for overfitting
best_score = grid_search.best_score_

cv_scores = grid_search.cv_results_['mean_test_score']
index_of_best_params = grid_search.best_index_
cv_score_of_best_params = cv_scores[index_of_best_params]

# Check feature importance 
best_model_coef_table = pd.DataFrame(columns=['Feature', 'Importance'])

for feature, coef in zip(best_model.feature_importances_, X_train.columns):
    best_model_coef_table = pd.concat([best_model_coef_table, pd.DataFrame({'Feature': [coef], 'Importance': [feature]})], ignore_index=True)

####################################################################################################
####################################################################################################
####################################################################################################

submission = pd.DataFrame({
                        "PassengerId":test['PassengerId'],
                        "Survived": pred
                        })

submission.to_csv(path + '/submission.csv', index=False)

####################################################################################################
####################################################################################################
####################################################################################################
