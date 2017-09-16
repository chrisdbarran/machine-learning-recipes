from scipy.spatial import distance

# Define the function to return the distance between two points
def euc(a, b):
    return distance.euclidean(a, b)

class ScrappyKNN():
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    # Find the closest point to make a prediction
    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        # compare to all other points in the training data.
        # to find the closest point then return the value of
        # the closest point as the prediction
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

x = iris.data
y = iris.target

# Split up a data set between testing and training. (.5 = 50:50 split)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .5)

# Use a decision tree classifier to build a model
from sklearn import tree
my_classifier = ScrappyKNN()

my_classifier.fit(x_train, y_train)

# Generate predictions on the test set
predictions = my_classifier.predict(x_test)

# Calculate the accuracy of the predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
