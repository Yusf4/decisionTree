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
x_min,x_max=data[:,0].min()-1,data[:,0].max()+1
y_min,y_max=data[:, 1].min()-1,data[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))

Z=tree_model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx,yy,Z,alpha=0.8,cmap=plt.cm.Paired)
plt.scatter(data[:,0],data[:,1],c=labels,edgecolors='k',cmap=plt.cm.Paired)
plt.title("decision tree decision boundary")
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
print(f"Accuracy :{accuracy:.2f}")
print(f"confusion Matrix:")
print(conf_matrix)
print("\n classification report:")
print(report)
