import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#Generate the dataSet
data ,labels=make_classification(
    n_samples=500, n_features=2,n_classes=2,n_informative=2,
    n_redundant=0,random_state=42,n_clusters_per_class=1
)
# split dataSet
X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.3,random_state=42)
#initialize and fit the decision tree model
tree_model= DecisionTreeClassifier(max_depth=3,random_state=42)
tree_model.fit(X_train,y_train)
#Make predictons
predictions=tree_model.predict(X_test)
#Evaluate the model
accuracy=accuracy_score(y_test,predictions)
conf_matrix=confusion_matrix(y_test,predictions)
report=classification_report(y_test,predictions)
print(f"Accuracy :{accuracy:.2f}")
print(f"confusion Matrix:")
print(conf_matrix)
print("\n classification report:")
print(report)
