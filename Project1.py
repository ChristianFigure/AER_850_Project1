import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits import mplot3d

# -------------------- INTRODUCTION --------------------
    # Predictive machine learning (ML) algorithms are being increasingly studied in 
    # augmented reality (AR) training modules. Particularly in the manufacturing and maintenance 
    # sectors of the aerospace industry, this study field has tremendous prospects to improve the 
    # effectiveness and precision of AR modules. A potent strategy for utilizing ML models to 
    # improve current AR-based animation approaches is to use a micro-level coordinate system 
    # approach. By combining these models with modern technology, personnel can obtain greater 
    # understanding while carrying out a range of tasks, including routine maintenance for space-
    # related applications, computer-aided design and modeling, part assembly, and airplane maintenance.

# -------------------- THEORY --------------------
    # In order to accurately forecast the maintenance step or stage connected to a certain part 
    # and its accompanying coordinates, this research seeks to examine various machine learning 
    # algorithms based on categorization approaches. The inverter from the FlightMax Fill Motion 
    # Simulator is the exact component being considered for this project.
    # Once the chosen component has been identified, the project specifies a total of 13 unique 
    # steps that make up the inverter disassembly procedure, each of which is identified by accurate X,
    # Y, and Z axis coordinates. These coordinates make up the input features for this ML model, and 
    # the target variable is the specific maintenance step to which these coordinates apply.


# -------------------- RESULTS --------------------
    
# ----------Data Processing----------
df = pd.read_csv("Project1Data.csv")

# ----------Data Visualization----------
df.plot(kind ='line')
SummaryDataFrame=df.describe()
print(SummaryDataFrame)
    # As shown in the plot line figure, we have a Discrete Numerical Data of 3D coordinates of X,Y,Z, and a Step.
    # It appears that the Step has an increasing trend with multiple steps at the first and last sample data. It
    # appears that the Step will increase regardless of the X,Y, and Z coordinates but we will analyze their 
    # relationship further in the code, whether each coordinate correlates with it.
    # Furthermore, we can extract and summarize more information from the Data Frame by using the describe function.
    # It was found that the number of samples for each data is 860. The X coordinate ranges from 0 to 9.375, the Y
    # coordinate ranges from 3.0625, the Z coordinates range from 0 to 2.35, and the Step range from 1 to 13. When
    # preding the Step, we should expect that it will fall within this range. Also, the Z was determined to have
    # the lowest std, and the X has the highest std, these means that the X coordinates has a relative high variation
    # while the Z varies less.

# ----------Correlation Analysis----------
cor = df.corr()
print(cor)
sns.heatmap(cor, annot = True, cmap = "mako")
    # To further understand the relationship of the coordinates with the step, we will used a table called correlation matrix.
    # which is defined by the corr() function. This function describes the statistical relationship and dependency between
    # the coordinates and step. As shown in the correlation matrix X and Z, X and Y, Y and Z, Y and Step, and Z and Step has a magnitude 
    # between 0.1 and 0.3 which means they hahve a very weak correlation. X and Step on the other hand, have a -0.75
    # correlation strength which means they have a stong correlation.
    # The impact of this is with high magnitude of correlation, it is likely to have a strong influence on our prediction
    # which can lead to more accurate prediction. In this case X has a higher influence on the Step.
    # Y and Z on the other hand can also influence the steps but not as much as X.

# ----------Classification Model Development/Engineering----------
X = df.drop('Step', axis = 1)
y = df['Step']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
    # Now that we are done analyzing the correlation matrix, we are ready to create our model by training and testing our data.
    # As shown in the line of code above, our test size is 0.2 which means that our test size is 20 percent of the given
    # number of data and the 80 percent will be our training set. Also, Random state is chosen to be 42, which is just a random number
    # but for this project we will be using 42 to use the same random sample throughout the modelling.
    # Furthermore, in order to avoid any type of biases, we will scale both the X_train and the X_test. For the models,
    # I have chosen the Decision Tree Classifier, Random Forest Classifier, and SVM CLassifier. The first model is Linear Regression,
    # which is simple yet powerful model to predict data. For this specific project, we are predicting 13 unique steps and
    # Decision Tree Classifier is a very simple and straight forward model which can predict non linear models easily, 
    # however, due to its simplicity it is susceptible to overfitting but we will still consider it due to its straightforwadness.
    # The second model is Random Forest Classifier which is an ideal model because of the amount of unique steps we are tryin to predict,
    # and Random Forest Classifier is a good candidate for this and it is less prone to overfitting, 
    # though its required a little bit more powerful computer. Lastly, SVM model can handle mix of numerical coordinates,
    # which is also ideal in this case and it is also less prone to overfitting, thought its hyperparameters are really
    # sensitive.
    
#Decision Tree Classifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train, y_train)
pred_dtc = dtc.predict(X_test)
# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=30, random_state=42)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
#SVM Classifier
clf=svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)

# GridSearchCV
# Decision Tree Classifier 
param_grid_dtc= {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search_dtc = GridSearchCV(dtc, param_grid_dtc, cv=5, n_jobs=-1)
grid_search_dtc.fit(X_train, y_train)
best_params_dtc = grid_search_dtc.best_params_
print("Best Hyperparameters for dtc:", best_params_dtc)

# Random Forest Classifier
param_grid_rfc= {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search_rfc = GridSearchCV(rfc, param_grid_rfc, cv=5, n_jobs=-1)
grid_search_rfc.fit(X_train, y_train)
best_params_rfc = grid_search_rfc.best_params_
print("Best Hyperparameters for rfc:", best_params_rfc)

#SVM Classifier
param_grid_clf= {
    'C': [0.1,1,100,1000],
    'kernel': ['rbf','poly','sigmoid','linear'],
    'degree': [1,2,3,4,5,6],
}
grid_search_clf = GridSearchCV(clf, param_grid_clf, cv=5, n_jobs=-1)
grid_search_clf.fit(X_train, y_train)
best_params_clf = grid_search_clf.best_params_
print("Best Hyperparameters for clf:", best_params_clf)

    # Now that we have info about the best parameters for each model we are ready to cross validate.
    # The best Hyperparameters yields:
    # Best Hyperparameters for dtc: {'criterion': 'gini', 'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 4, 'min_samples_split': 5}
    # Best Hyperparameters for rfc: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    # Best Hyperparameters for clf: {'C': 100, 'degree': 5, 'kernel': 'poly'}
best_dtc = DecisionTreeClassifier(criterion = 'gini', max_depth= None, max_features= 'log2', min_samples_leaf= 4, min_samples_split= 5)
best_rfc = RandomForestClassifier(max_depth= None, max_features= 'sqrt', min_samples_leaf= 1, min_samples_split= 2, n_estimators= 100) 
best_clf = svm.SVC(C= 100, degree= 5, kernel= 'poly')   
cross_val_dtc=cross_val_score(best_dtc, X_train, y_train, cv=5,scoring='accuracy').mean()
print(cross_val_dtc)
cross_val_rfc=cross_val_score(best_rfc, X_train, y_train, cv=5,scoring='accuracy').mean()
print(cross_val_rfc)
cross_val_clf=cross_val_score(best_clf, X_train, y_train, cv=5,scoring='accuracy').mean()
print(cross_val_dtc)
    # Based on the Grid Search Cross Validation results, the score for Decision Tree Classifier is about 98.54%,
    # the Random Forest Classifier Scores 99.56%, and SVM Classifier scores 98.54%. As you can see the models
    # have a really high cross validation score, where they are all more than 98%, so regardless of which model we
    # pick it will perform very well on the data as they can accurately assess and predict unseen data. 
    # But since the higher the better, the best model is the
    # Random Forest Classifier which we will be using for the maintenance step provided in the lab manual.
    
# ----------Model Performance Analysis----------   
# f1-score, precision, and accuracy
print(classification_report(y_test, pred_dtc))
print(classification_report(y_test, pred_rfc))
print(classification_report(y_test, pred_clf))
    # First we have to define what these value means. Accuracy is the measurement how accurate the categories are, including
    # both true positives and true negatives. It calculates the overall correctness of the data. However, this can be misleading
    # for imbalanced datasets. The precision is the percentage of positive predictions because it can quantify the models
    # ability to avoid false positive error. Finally, the f1-score is just the mean of precision and recall, this indicates
    # that higher recall and precision value yields to higher f1-score.
    # Based on the code above the prediction for Decision Tree Classifier for Steps 1 to 13 have a weighted average of 99%
    # for f1-score, precision, and accuracy. The Random Forest Classifier have a weighted average of 98 % for f1-score,
    # precision, and accuracy. Lastly, the SVM Classfier have a weight average of 100% for the precision, and 99% for
    # both the f1-score, and accuracy. 
    # For accuracy we want to see the output variable first which in this case Step. So we have to determined
    # whether the Step data is balance or imbalance. This can be check with the line of code below
# print(df.hist())
    # As shown in the histogram, the step data is skewed and not normally distributed meaning the step data set is imbalance.
    # As mentioned before, accuracy is highly realiable when the data set is balanced, but since the step data set is not
    # normally distributed, we have to pick the lowest accuracy to avoid misleading prediction which in this case the
    # Random Forest Classifier. For precision, we want to ensure that positive predictions are highly accurate, hence, the
    # higher the precision score the better which in this case the CVM CLassifier.
    # For the f1-score, since we have imbalance in our step data set which is the desired value, it makes more sense to
    # select the model that got the highest f-score which in this case both the Decision Tree Classifier and SVM Classifier.
    # Overall, these models have an extremely high f1-score, precision, and accuracy, and we can select any of them with no
    # problem. But we will stick to the Random Forest CLassifier. The confusion matrix is shown in the code below.
plt.figure(figsize=(10,7))
scores_rfc= cross_val_score(rfc, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
sns.heatmap(confusion_matrix(y_test,pred_rfc), annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# ----------Model Evaluation----------
import joblib 
joblib.dump(rfc,"rfc")
rfc_=joblib.load("rfc")
print(rfc_.predict(sc.transform([[9.375,3.0625,1.51]])))
print(rfc_.predict(sc.transform([[6.995,5.125,0.3875]])))
print(rfc_.predict(sc.transform([[0,3.0625,1.93]])))
print(rfc_.predict(sc.transform([[9.4,3,1.8]])))
print(rfc_.predict(sc.transform([[9.4,3,1.3]])))
    #  As shown from the code above the prediction steps for the given coordinates in the that order are:
    #  5, 8, 13, 6, and 4.

# -------------------- CONCLUSION --------------------
    # In the end, the purpose of this project is to explore various classification-based machine learning which includes:
    # Decision Tree Classifier, Random Forest Classifier, SVM Classifier. After training and testing the data with these models considering
    # some factors such as accuracy, precision, and f1-score, the chosen model was Random Forest Classifier.
    # Finally it was found that the provided sample coordinates have a step of 5, 8, 13, 6, and 4.
