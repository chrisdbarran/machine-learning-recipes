from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

# Split up a data set between testing and training. (.5 = 50:50 split)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# Use a decision tree classifier to build a model
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(x_train, y_train)

# Generate predictions on the test set
predictions = my_classifier.predict(x_test)

# Calculate the accuracy of the predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

# Use a next nearest neighbours classifier

from sklearn.neighbors import KNeighborsClassifier
kn_classifier = KNeighborsClassifier()
kn_classifier.fit(x_train, y_train)
kn_predictions = kn_classifier.predict(x_test)

print(accuracy_score(y_test,kn_predictions))