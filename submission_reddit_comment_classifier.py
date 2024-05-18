import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sqlalchemy import create_engine

# Ensure necessary NLTK data packages are downloaded
nltk.download('stopwords')
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

    # Step 3: Split data into training and testing sets
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Step 4: Preprocess the Training Data
    df_train['cleaned_comments'] = df_train['comments'].apply(preprocess_text)

    # Load manually labeled training data from a CSV file
    labeled_data = pd.read_csv('labeled_reddit_comments.csv')

    # Preprocess the labeled data
    labeled_data['cleaned_comments'] = labeled_data['comments'].apply(preprocess_text)

    # Step 5: Combine the labeled data with the training set
    df_train = pd.concat([df_train, labeled_data])

    # Step 6: Feature Engineering
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train = vectorizer.fit_transform(df_train['cleaned_comments'])
    X_test = vectorizer.transform(df_test['comments'].apply(preprocess_text))
    y_train = df_train['label']
    y_test = df_test['label']

    # Step 7: Model Selection and Training
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Step 8: Evaluate the Model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Step 9: Classify the Remaining Data
    df_test['label'] = model.predict(X_test)

    # Save the classified data to a CSV file
    df_test.to_csv('classified_reddit_comments.csv', index=False)

if __name__ == '__main__':
    main()
