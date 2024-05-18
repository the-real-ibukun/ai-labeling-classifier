import openai
import pandas as pd
from sqlalchemy import create_engine

# Set your OpenAI API key
openai.api_key = ''

# Database credentials
db_params = {
    'dbname': 'Vetassist',
    'user': 'niphemi.oyewole',
    'password': 'W7bHIgaN1ejh',
    'host': 'ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech',
    'port': '5432',
    'endpoint_id': 'ep-delicate-river-a5cq94ee'
}

def classify_comment(comment):
    prompt = (
        f"Label the following comment as either 'Medical Doctor', 'Veterinarian', or 'Other':\n"
        f"Comment: \"{comment}\"\n"
        f"Label: "
    )
    
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=10,
        temperature=0
    )
    
    label = response.choices[0].text.strip()
    return label

def retrieve_and_label_data():
    # Create the connection URL
    connection_url = (
        f"postgresql://{db_params['user']}:{db_params['password']}@"
        f"{db_params['host']}:{db_params['port']}/{db_params['dbname']}?"
        f"options=endpoint%3D{db_params['endpoint_id']}&sslmode=require"
    )
    
    # Create the database engine
    engine = create_engine(connection_url)
    
    # Query to retrieve data
    query = "SELECT username, comments FROM public.reddit_usernames_comments"
    
    # Read the data into a DataFrame
    df = pd.read_sql_query(query, engine)
    
    # Apply the classification function to each comment
    df['label'] = df['comments'].apply(classify_comment)
    
    # Save the labeled data to a CSV file
    df.to_csv('labeled_reddit_comments.csv', index=False)
    print("Labeled data has been saved to 'labeled_reddit_comments.csv'")

if __name__ == '__main__':
    retrieve_and_label_data()
