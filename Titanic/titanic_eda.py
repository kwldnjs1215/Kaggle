import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

path = '/content/gdrive/MyDrive/Kaggle/Titanic'

train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')
gender_sub = pd.read_csv(path + '/gender_submission.csv')

############## NA ##############

# train
msno.matrix(train, figsize=(12, 5))
train.isnull().sum()

# test
msno.matrix(test, figsize=(12, 5))
test.isnull().sum()

############## Survived ##############
survived_counts = train["Survived"].value_counts(sort=False)

plt.figure(figsize=(8, 5))
plt.pie(survived_counts, labels=['Dead', 'Survived'], autopct='%1.1f%%')
plt.title('Titanic Survival Ratio')
plt.show()

############## Sex ##############
# Sex
sex_counts = train["Sex"].value_counts(sort=False)

plt.figure(figsize=(8, 5))
plt.pie(sex_counts, labels=['Male', 'Female'], autopct='%1.1f%%')
plt.title('Sex Ratio')
plt.show()

# Sex, Survived
sex_group = train.groupby(["Sex", "Survived"], sort=False).size().unstack()

sex_group.plot(kind='bar', stacked=False)
plt.title('Survival Counts by Sex')
plt.xlabel('')
plt.ylabel('Count')
plt.xticks([0, 1], ['Male', 'Female'])
plt.legend(labels=['Dead', 'Survived'])

############## Pclass ##############
# Pclass
class_count = train["Pclass"].value_counts().sort_index()

class_count.plot(kind='bar')
plt.title('Survival Rate by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.legend()

# Pclass, Survived
class_group = train.groupby(["Pclass", "Survived"]).size().unstack().sort_index()

class_group.plot(kind='bar', stacked=False)
plt.title('Survival Counts by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.xticks([0, 1, 2], ['1', '2', '3'])
plt.legend(labels=['Dead', 'Survived'])

############## Age ##############
# Age
plt.figure(figsize=(8, 5))
sns.distplot(train['Age'], bins=25)

# Age, Survived
dead = train[train['Survived'] == 0]
sur = train[train['Survived'] == 1]

plt.figure(figsize=(8, 5))
sns.distplot(dead['Age'], bins=25, hist=False, label='Dead')
sns.distplot(sur['Age'], bins=25, hist=False, label='Survived')
plt.legend()
plt.show()

############## SibSp ##############
# SibSp
sibsp_count = train["SibSp"].value_counts().sort_index()

sibsp_count.plot(kind='bar')
plt.title('Survival Rate by SibSp')
plt.xlabel('SibSp')
plt.ylabel('Count')
plt.legend()

# SibSp, Survived
sibsp_group = train.groupby(["SibSp", "Survived"]).size().unstack().sort_index()

sibsp_group.plot(kind='bar')
plt.title('Survival Counts by SibSp')
plt.xlabel('SibSp')
plt.ylabel('Count')
plt.legend(labels=['Dead', 'Survived'])

############## Parch ##############
# Parch
parch_count = train["Parch"].value_counts().sort_index()

parch_count.plot(kind='bar')
plt.title('Survival Rate by Parch')
plt.xlabel('Parch')
plt.ylabel('Count')
plt.legend()

# Parch, Survived
parch_group = train.groupby(["Parch", "Survived"]).size().unstack().sort_index()

parch_group.plot(kind='bar')
plt.title('Survival Counts by Parch')
plt.xlabel('Parch')
plt.ylabel('Count')
plt.legend(labels=['Dead', 'Survived'])

############## Embarked ##############
# Embarked
embarked_count = train["Embarked"].value_counts().sort_index()

embarked_count.plot(kind='bar')
plt.title('Survival Rate by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.legend()

# Embarked, Survived
embarked_group = train.groupby(["Embarked",  "Survived"]).size().unstack().sort_index()

embarked_group.plot(kind='bar')
plt.title('Survival Counts by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Count')
plt.legend(labels=['Dead', 'Survived'])

# Pclass, Embarked
pc1 = train[train['Pclass'] == 1]['Embarked'].value_counts().sort_index()
pc2 = train[train['Pclass'] == 2]['Embarked'].value_counts().sort_index()
pc3 = train[train['Pclass'] == 3]['Embarked'].value_counts().sort_index()

pc = pd.DataFrame([pc1, pc2, pc3], index=['Pclass 1', 'Pclass 2', 'Pclass 3'])

pc.plot(kind='bar', stacked=True)
plt.title('Embarked Counts by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Embarked')
plt.show()

############## Fare ##############
fig, ax = plt.subplots(figsize=(10,6))

sns.kdeplot(train[train['Survived']==0]['Fare'], ax=ax)
sns.kdeplot(train[train['Survived']==1]['Fare'], ax=ax)

ax.set(xlim=(0, train['Fare'].max()))
ax.legend(['Dead', 'Survived'])

plt.show()

