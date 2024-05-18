import openai
import pandas as pd
from sqlalchemy import create_engine
import time
from sklearn.model_selection import train_test_split

# Set your OpenAI API key
openai.api_key = ""

def generate_text_labels(texts, categories):
    labels = []
    text_label_mapping = {}

    # String of categories in which you want to classify the text.
    category_str = ", ".join(map(str, categories))
    
    for text in texts:
        if len(text) > 16140:
            label = "Comment too long"
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"{text}; Classify this sentence as {category_str} in one word."},
                ]
            )
            label = response.choices[0]["message"]["content"].strip(".")
        
        labels.append(label)
        text_label_mapping[text] = label
    
    return labels, text_label_mapping

def retrieve_data(start_index=0):
    # Database credentials
    db_params = {
        'dbname': 'Vetassist',
        'user': 'niphemi.oyewole',
        'password': 'W7bHIgaN1ejh',
        'host': 'ep-delicate-river-a5cq94ee-pooler.us-east-2.aws.neon.tech',
        'port': '5432',
        'endpoint_id': 'ep-delicate-river-a5cq94ee'
    }

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

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    labeled_comments = []
    num_comments = len(train_df)
    
    categories = ['Medical Doctor', 'Veterinarian', 'Other']

    for i, (_, row) in enumerate(train_df.iterrows()):
        if i < start_index:
            continue
        
        username = row['username']
        comment = row['comments']
        labels, _ = generate_text_labels([comment], categories)
        label = labels[0]
        
        labeled_comments.append({'username': username, 'comments': comment, 'label': label})
        
        # Print a message for each comment labeled
        print(f"Comment {i + 1} labeled.")
        
        # Export labeled data after labeling each comment
        labeled_df = pd.DataFrame(labeled_comments)
        labeled_df.to_csv('finals_reddit_comments_data_labeling.csv', index=False)
        
        time.sleep(0.5)  # Optional: add a delay between API requests to avoid rate limits

    # Save the remaining labeled data to a CSV file
    labeled_df = pd.DataFrame(labeled_comments)
    labeled_df.to_csv('finals_reddit_comments_data_labeling.csv', index=False)
    print("Data for labeling has been saved to 'finals_reddit_comments_data_labeling.csv'")

if __name__ == '__main__':
    # You can specify the index from which to resume labeling
    retrieve_data(start_index=0)
