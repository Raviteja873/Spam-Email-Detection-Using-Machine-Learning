import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
df = pd.read_csv("spam_ham_dataset.csv")

# 2. Count spam vs ham
num_spam = df[df['label'] == 'spam'].shape[0]
num_ham = df[df['label'] == 'ham'].shape[0]

print(f"📨 Total spam emails: {num_spam}")
print(f"📨 Total ham (not spam) emails: {num_ham}")

# 3. Use 'text' as feature and 'label_num' as target
X = df['text']
y = df['label_num']  # 0 = ham, 1 = spam

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Vectorize the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Predict
y_pred = model.predict(X_test_vec)

# 8. Evaluate
print("\n Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
