
# Loading the training and test dataset
reviews_train = []
for line in open("aclImdb/movie_data/full_train.txt", "r"):
    reviews_train.append(line.strip())

reviews_test = []
for line in open("aclImdb/movie_data/full_test.txt", "r"):
    reviews_test.append(line.strip())

# Removing special characters
import re

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]

    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

# Vectorization / One Hot Encoding
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

# Build classifier and use validation set to tune c
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# First half of the reviews are positive, the second half negative
target = [1 if i < 12500 else 0 for i in range(25000)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:

    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s"
           % (c, accuracy_score(y_val, lr.predict(X_val))))

# Accuracy for C=0.01: 0.87264
# Accuracy for C=0.05: 0.88448 <-- best accuracy
# Accuracy for C=0.25: 0.87984
# Accuracy for C=0.5: 0.87712
# Accuracy for C=1: 0.87296

# Training the final model

final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
print ("Final Accuracy: %s"
       % accuracy_score(target, final_model.predict(X_test)))
