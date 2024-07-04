from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Route to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("app/static/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# Function to load data from CSV
def load_data_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df = df.fillna('')  # Fill NaN values with empty strings
        
        # Ensure the 'id' column is present and properly assigned
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)  # Generate IDs from 1 to len(df)
            logging.info("Generated IDs for the dataset.")
        
        # Ensure the ID column is a string
        df['id'] = df['id'].astype(str)
        
        logging.debug(f"Loaded {len(df)} data with columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

# Detect relevant columns
def detect_columns(df):
    # Define potential column names for titles
    potential_title_cols = ['title', 'movie_title', 'article_title', 'book_title', 
        'Series_Title', 'name', 'headline', 'heading', 'názov', 'titulok']
    title_col = next((col for col in df.columns if any(potential in col.lower() for potential in potential_title_cols)), None)
    
    # Define potential column names for content or text
    potential_content_cols = ['text', 'content', 'description', 'plot', 'overview', 
        'body', 'summary', 'details', 'popis', 'obsah']
    content_col = next((col for col in df.columns if any(potential in col.lower() for potential in potential_content_cols)), None)
    
    if not title_col or not content_col:
        raise HTTPException(status_code=400, detail="Required columns ('title' and 'description') not found in the dataset")
    
    return 'id', title_col, content_col

# Precompute TF-IDF vectors and cosine similarity matrix
def compute_similarity_matrix(df):
    id_col, title_col, content_col = detect_columns(df)
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[content_col])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return cosine_sim_matrix, id_col, title_col, content_col

# Global variables to store data and similarity matrix
data_df = pd.DataFrame()  # Empty DataFrame initially
cosine_sim_matrix = None
id_col = None
title_col = None
content_col = None

# Endpoint to upload CSV file
@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    global data_df, cosine_sim_matrix, id_col, title_col, content_col
    
    try:
        # Save the uploaded file locally
        file_path = os.path.join("app", "data", file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Load data from the uploaded CSV
        data_df = load_data_from_csv(file_path)
        
        # Compute similarity matrix
        cosine_sim_matrix, id_col, title_col, content_col = compute_similarity_matrix(data_df)
        
        # Return success message
        return {"detail": f"Uploaded {file.filename} successfully and processed data."}
    
    except Exception as e:
        logging.error(f"Error uploading and processing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading and processing data: {str(e)}")

# Endpoint for searching data
@app.get("/search/")
async def search_data(query: str):
    global data_df, id_col, title_col, content_col
    
    if data_df.empty:
        raise HTTPException(status_code=404, detail="No data available to search. Upload a CSV file first.")
    
    logging.debug(f"Searching for data with query: {query}")
    
    # Tokenize the query
    query_tokens = query.lower().split()
    
    # Function to check if all query tokens are in the title
    def matches_query(title):
        title_lower = title.lower()
        return all(token in title_lower for token in query_tokens)
    
    # Filter data based on title matching
    results = data_df[data_df[title_col].apply(matches_query)]
    
    if results.empty:
        logging.warning(f"No data found matching query: {query}")
        raise HTTPException(status_code=404, detail="No data found matching the query")
    
    # Return only the id and title columns in the search results
    return {"results": results[[id_col, title_col]].to_dict(orient='records')}

# Endpoint for generating recommendations based on content similarity
@app.get("/recommendations/")
async def get_recommendations(id: str, num_recommendations: int = Query(5, ge=1, le=20)):
    global data_df, cosine_sim_matrix, id_col, title_col, content_col
    
    if data_df.empty:
        raise HTTPException(status_code=404, detail="No data available to recommend. Upload a CSV file first.")
    
    logging.debug(f"Generating recommendations for id: {id} with num_recommendations: {num_recommendations}")
    
    # Ensure ID is a string
    id = str(id)
    
    # Validate if the provided id exists in the dataset
    if id not in data_df[id_col].values:
        logging.error(f"ID {id} not found in the dataset")
        raise HTTPException(status_code=404, detail="ID not found in the dataset")
    
    # Get the index of the item
    idx = data_df.index[data_df[id_col] == id].tolist()[0]
    logging.debug(f"Found index {idx} for id {id}")
    
    # Compute similarity scores based on descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_df[content_col])
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get indices of top similar items
    similarity_scores = list(enumerate(cosine_sim_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_data_indices = [i[0] for i in similarity_scores[1:num_recommendations + 1]]  # Exclude itself
    
    recommendations = data_df.iloc[similar_data_indices]
    
    if recommendations.empty:
        logging.error("No recommendations available")
        raise HTTPException(status_code=404, detail="No recommendations available")
    
    # Prepare recommendations with IDs
    recommended_items = []
    for _, row in recommendations.iterrows():
        recommended_items.append({
            "id": row[id_col],
            "title": row[title_col],
            "content": row[content_col]
        })
    
    logging.debug("Returning recommendations")
    return {"recommendations": recommended_items}

# Endpoint to get item details by ID
@app.get("/item/{item_id}", response_model=dict)
async def get_item(item_id: str):
    global data_df, id_col, title_col, content_col
    
    if data_df.empty:
        raise HTTPException(status_code=404, detail="No data available. Upload a CSV file first.")
    
    logging.debug(f"Fetching item with ID: {item_id}")
    try:
        item = data_df[data_df[id_col] == item_id].iloc[0].to_dict()
        logging.debug(f"Found item: {item}")
        return {"item": item}
    except IndexError:
        logging.error(f"Item with ID {item_id} not found")
        raise HTTPException(status_code=404, detail=f"Item with ID {item_id} not found")
