import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sqlalchemy import create_engine

# Ensure necessary NLTK data packages are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Database credentials
db_params = {
    'dbname': 'Vetassist',
    'user': 'niphemi.oyewole',
    'password': 'W7bHIgaN1ejh',
    'host': 'ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech',
    'port': '5432',
    'endpoint_id': 'ep-delicate-river-a5cq94ee'
}

def preprocess_text(text):
    # Tokenize, remove stop words, lower case
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [word.lower() for word in tokens if word.isalpha() and word not in stopwords]
    return ' '.join(tokens)

def main():
    # Step 1: Connect to the Database using SQLAlchemy
    connection_url = (
        f"postgresql://{db_params['user']}:{db_params['password']}@"
        f"{db_params['host']}:{db_params['port']}/{db_params['dbname']}?"
        f"options=endpoint%3D{db_params['endpoint_id']}&sslmode=require"
    )
    engine = create_engine(connection_url)

    # Step 2: Retrieve Data
    query = "SELECT username, comments FROM public.reddit_usernames_comments"
    df = pd.read_sql_query(query, engine)

    # Step 3: Load manually labeled training data from a CSV file
    labeled_data = pd.read_csv('train_set2_labeled_reddit_comments.csv')

    # Filter labeled data to include only the desired labels
    desired_labels = ['Medical Doctor', 'Veterinarian', 'Other']
    labeled_data = labeled_data[labeled_data['comments'].isin(desired_labels)]

    # Step 4: Combine the labeled data with the retrieved data
    df = pd.concat([df, labeled_data])

    # Step 5: Preprocess the comments
    df['cleaned_comments'] = df['comments'].apply(preprocess_text)

    # Split data into training and testing sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Separate features and labels
    X_train = df_train['cleaned_comments']
    y_train = df_train['comments']
    X_test = df_test['cleaned_comments']
    y_test = df_test['comments']

    # Step 6: Feature Engineering
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Step 7: Model Selection and Training
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    # Step 8: Evaluate the Model
    y_pred = model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, labels=desired_labels))

    # Step 9: Classify the Remaining Data
    df_test['predicted_label'] = model.predict(X_test_vec)

    # Save the classified data to a CSV file
    df_test.to_csv('classified_reddit_comments.csv', index=False)

if __name__ == '__main__':
    main()
