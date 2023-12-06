import pandas as pd
import gradio as gr
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import requests
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from fuzzywuzzy import fuzz
warnings.filterwarnings("ignore")

model = SentenceTransformer('all-mpnet-base-v2') 
df1 = pd.read_parquet('df_encoded.parquet')
df1 = df1.rename(columns={"Description": "Products:"})
df1 = df1.rename(columns={"UnitPrice": "Price:"})
nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(df1['text_vector_'].values.tolist())

movies_list = pickle.load(open('movie_dict.pkl','rb'))
movies = pd.DataFrame(movies_list)
similarity1=pickle.load(open('similarities.pkl','rb'))
listing = movies['title'].values
listing = listing.tolist()

df2 = pickle.load(open('final_ratings.pkl','rb'))
books = pd.DataFrame(df2)

df3 = pd.read_csv("steamdeploy.csv",  error_bad_lines=False, encoding='utf-8')

sim_matrix = pickle.load(open('gamefinal.pkl','rb'))
def matching_score(a,b):
  #fuzz.ratio(a,b) calculates the Levenshtein Distance between a and b, and returns the score for the distance
   return fuzz.ratio(a,b)
def get_title_year_from_index(index):
   return df3[df3.index == index]['year'].values[0]
#Convert index to title
def get_title_from_index(index):
   return df3[df3.index == index]['name'].values[0]
#Convert index to title
def get_index_from_title(title):
   return df3[df3.name == title].index.values[0]
#Convert index to score
def get_score_from_index(index):
   return df3[df3.index == index]['score'].values[0]
#Convert index to weighted score
def get_weighted_score_from_index(index):
   return df3[df3.index == index]['weighted_score'].values[0]
#Convert index to total_ratings
def get_total_ratings_from_index(index):
   return df3[df3.index == index]['total_ratings'].values[0]
#Convert index to platform
def get_platform_from_index(index):
  return df3[df3.index == index]['platforms'].values[0]
def get_photo_from_index(index):
  return df3[df3.index == index]['header_image'].values[0]

def find_closest_title(title):
   leven_scores = list(enumerate(df3['name'].apply(matching_score, b=title))) 
   sorted_leven_scores = sorted(leven_scores, key=lambda x: x[1], reverse=True)
   closest_title = get_title_from_index(sorted_leven_scores[0][0])
   distance_score = sorted_leven_scores[0][1]
   return closest_title, distance_score

gamelisting = pickle.load(open('gamelist3.pkl','rb'))

def recommendg(game, platform):
  #Return closest game title match
  closest_title, distance_score = find_closest_title(game)
  #Create a Dataframe with these column headers
  recomm_df = pd.DataFrame()
  #find the corresponding index of the game title
  games_index = get_index_from_title(closest_title)
  #return a list of the most similar game indexes as a list
  games_list = list(enumerate(sim_matrix[int(games_index)]))
  #Sort list of similar games from top to bottom
  similar_games = list(filter(lambda x:x[0] != int(games_index), sorted(games_list,key=lambda x:x[1], reverse=True)))
  n_games = []
  for i,s in similar_games:
    if platform in get_platform_from_index(i):
      n_games.append((i,s))
  #Only return the games that are above the minimum score
    
  #Return the game tuple (game index, game distance score) and store in a dataframe
  for i,s in n_games[:5]: 
    #Dataframe will contain attributes based on game index
    row = [get_title_from_index(i), get_title_year_from_index(i),  get_score_from_index(i), 
        get_weighted_score_from_index(i), 
        get_total_ratings_from_index(i),  get_photo_from_index(i)]
    recomm_df = recomm_df.append(row, ignore_index = True)
    a = recomm_df.values
    products_list = tuple(a.reshape(1, -1)[0])

  #Only include games released same or after minimum year selected
  return products_list[0], products_list[6], products_list[12], products_list[18], products_list[24], products_list[5], products_list[11], products_list[17], products_list[23], products_list[29], products_list[1], products_list[7], products_list[13], products_list[19], products_list[25],products_list[2], products_list[8], products_list[14], products_list[20], products_list[26] , products_list[3], products_list[9], products_list[15], products_list[21], products_list[27], products_list[4], products_list[10], products_list[16], products_list[22], products_list[28]

def generate_pt():
    final_ratings = pickle.load(open('final_ratings.pkl', 'rb'))
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    return pt

def correct (link):
    newlink = link.replace("http://images.amazon.com", "https://images-na.ssl-images-amazon.com")
    newlink = newlink.replace("MZZZZZZZ", "LZZZZZZZ")
    return newlink

books = books.rename(columns={"Book-Title": "title"})
books = books.rename(columns={"Image-URL-M": "url"})
books = books.rename(columns={"Year-Of-Publication": "year"})
books = books.rename(columns={"Book-Author": "author"})
books = books.drop_duplicates(subset=['title'])
books = books.reset_index(drop=True)
pt = generate_pt()
similarity2 = cosine_similarity(pt)
booklisting = books['title'].values
booklisting = booklisting.tolist()
booklisting = [*set(booklisting)]
booklisting.sort()

def recommendb(book):
    index_of_book = books[books['title']==book].index[0]
    distances = similarity2[index_of_book]
    books_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_books=[]
    recommended_books_author=[]
    recommended_books_year=[]
    recommended_books_poster=[]
    for i in books_list:
        recommended_books.append(books.iloc[i[0]].title)
        recommended_books_author.append(books.iloc[i[0]].author)
        recommended_books_year.append(books.iloc[i[0]].year)
        recommended_books_poster.append(books.iloc[i[0]].url)
    return recommended_books[0],recommended_books[1],recommended_books[2],recommended_books[3],recommended_books[4],recommended_books_author[0],recommended_books_author[1],recommended_books_author[2],recommended_books_author[3],recommended_books_author[4],recommended_books_year[0],recommended_books_year[1],recommended_books_year[2],recommended_books_year[3],recommended_books_year[4],correct (recommended_books_poster[0]),correct (recommended_books_poster[1]),correct (recommended_books_poster[2]),correct (recommended_books_poster[3]),correct (recommended_books_poster[4])

def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=c23694c0f1d41750259867c044d5a3de&language=en-US'.format(movie_id))
    data=response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']

def search(df, query):
    product = model.encode(query).tolist()
    distances, indices = nbrs.kneighbors([product])
    return df.iloc[list(indices)[0]][['Products:', 'Price:']]

def greet(text1): 
    return search(df1, text1)

def recommend(movie):
    index_of_the_movie = movies[movies['title']==movie].index[0]
    distances = similarity1[index_of_the_movie]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommended_movies=[]
    recommended_movies_poster=[]
    for i in movies_list:
        movie_id=movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_poster.append(fetch_poster(movie_id))
    return recommended_movies[0],recommended_movies[1],recommended_movies[2],recommended_movies[3],recommended_movies[4],recommended_movies_poster[0],recommended_movies_poster[1],recommended_movies_poster[2],recommended_movies_poster[3],recommended_movies_poster[4]

def get_sentences_data():
    with open('sentences.pkl', 'rb') as f:
        sentences = pickle.load(f)
    return sentences

sentences = get_sentences_data()

def get_embeddings_data():
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

embeddings = get_embeddings_data()

def get_model():
    model1 = SentenceTransformer('all-MiniLM-L6-v2')
    return model1

model1 = get_model()

def recommendp(paper):
    recommended_paper=[]
    cosine_scores = util.cos_sim(embeddings, model1.encode(paper))
    top_similar_papers = torch.topk(cosine_scores,dim=0, k=5,sorted=True)
    for i in top_similar_papers.indices:
        recommended_paper.append(sentences[i.item()])
    recommended_paper
    return recommended_paper[0],recommended_paper[1],recommended_paper[2],recommended_paper[3],recommended_paper[4]

with gr.Blocks() as demo:
    with gr.Tab(label="Products Recommender System"):
        gr.Markdown(
    """
    # VTI CORPORATION
    ## Products Recommender System 🛍
    This system will recommend what products to buy based on your specific needs or even keywords.
    """)
        gr.Markdown(
    """
    ## INPUT
    Please type in your needs or some keywords describing it:
    """)
        inp1 = gr.Textbox(value='A Christmas present for my 5 years old daughter', label='What are you looking for?')
        btn1 = gr.Button("Recommend")
        gr.Markdown(
    """
    Recommeded products:
    """)
        with gr.Row():
            outproduct = gr.DataFrame(headers=["Products:", "Price:"], row_count=8)                      

    with gr.Tab(label="Movies Recommender System"):
        gr.Markdown(
    """
    # VTI CORPORATION
    ## Movies Recommender System 🍿
    This system will recommend what movies to watch next based on your recently watched movie.
    """)
        gr.Markdown(
    """
    ## INPUT
    Choose your recently watched movie:
    """)
        inp2 = gr.Dropdown(listing,  value ='Avatar', label="Recently watched:")
        btn2 = gr.Button("Recommend")
        gr.Markdown(
    """
    Recommeded movies:
    """)
        with gr.Row():
            outm1 = gr.Textbox(label="Movie name:")
            outm2 = gr.Textbox(label="Movie name:")
            outm3 = gr.Textbox(label="Movie name:")
            outm4 = gr.Textbox(label="Movie name:")
            outm5 = gr.Textbox(label="Movie name:")
        with gr.Row():
            outm6 = gr.Image(label="Movie poster")
            outm7 = gr.Image(label="Movie poster")
            outm8 = gr.Image(label="Movie poster")
            outm9 = gr.Image(label="Movie poster")
            outm10 = gr.Image(label="Movie poster")       

    with gr.Tab(label="Books Recommender System"):
        gr.Markdown(
    """
    # VTI CORPORATION
    ## Books Recommender System 📚
    This system will recommend what books to read next based on the recent book that you read.
    """)
        gr.Markdown(
    """
    ## INPUT
    Choose your recently read book:
    """)
        inp3 = gr.Dropdown(booklisting, value = "One Door Away from Heaven", label="Recently read:")
        btn3 = gr.Button("Recommend")
        gr.Markdown(
    """
    Recommeded books:
    """)
        with gr.Row():
            out1 = gr.Textbox(label="Book name:")
            out2 = gr.Textbox(label="Book name:")
            out3 = gr.Textbox(label="Book name:")
            out4 = gr.Textbox(label="Book name:")
            out5 = gr.Textbox(label="Book name:")
        with gr.Row():
            out6 = gr.Textbox(label="Author:")
            out7 = gr.Textbox(label="Author:")
            out8 = gr.Textbox(label="Author:")
            out9 = gr.Textbox(label="Author:")
            out10 = gr.Textbox(label="Author:")
        with gr.Row():
            out11 = gr.Textbox(label="Year of publication:")
            out12 = gr.Textbox(label="Year of publication:")
            out13 = gr.Textbox(label="Year of publication:")
            out14 = gr.Textbox(label="Year of publication:")
            out15 = gr.Textbox(label="Year of publication:")                        
        with gr.Row():
            out16 = gr.Image(label="Book cover")
            out17 = gr.Image(label="Book cover")
            out18 = gr.Image(label="Book cover")
            out19 = gr.Image(label="Book cover")
            out20 = gr.Image(label="Book cover")                             

    with gr.Tab(label="Scientific Papers Recommender System"):
        gr.Markdown(
    """
    # VTI CORPORATION
    ## Scientific Papers Recommender System 📝
    This system will recommend what scientific papers to research based on your specific needs or even keywords.
    """)
        gr.Markdown(
    """
    ## INPUT
    Please type in your needs or some keywords describing it:
    """)
        inp4 = gr.Textbox(value='Adventures in Financial Data Science', label='What are you researching?')
        btn4 = gr.Button("Recommend")
        gr.Markdown(
    """
    Recommeded scientific papers:
    """)
        with gr.Column():
            outp1 = gr.Textbox(label="Paper name:")
            outp2 = gr.Textbox(label="Paper name:")
            outp3 = gr.Textbox(label="Paper name:")
            outp4 = gr.Textbox(label="Paper name:")
            outp5 = gr.Textbox(label="Paper name:") 

    with gr.Tab(label="Games Recommender System"):
        gr.Markdown(
    """
    # VTI CORPORATION
    ## Games Recommender System 🎮
    This system will recommend what games to play next based on the recently played game.
    """)
        gr.Markdown(
    """
    ## INPUT
    Choose your recently played game:
    """)
        inp5 = [gr.Dropdown(gamelisting, label="Recently played:"), gr.inputs.Radio(['windows','linux','mac'], label="Platform:")]
        btn5 = gr.Button("Recommend")
        gr.Markdown(
    """
    Recommeded games:
    """)
        with gr.Row():
            outg1 = gr.Textbox(label="Game name:")
            outg2 = gr.Textbox(label="Game name:")
            outg3 = gr.Textbox(label="Game name:")
            outg4 = gr.Textbox(label="Game name:")
            outg5 = gr.Textbox(label="Game name:")
        with gr.Row():
            outg6 = gr.Image(label="Game cover")
            outg7 = gr.Image(label="Game cover")
            outg8 = gr.Image(label="Game cover")
            outg9 = gr.Image(label="Game cover")
            outg10 = gr.Image(label="Game cover")        
        with gr.Row():
            outg11 = gr.Textbox(label="Year released")
            outg12 = gr.Textbox(label="Year released")
            outg13 = gr.Textbox(label="Year released")
            outg14 = gr.Textbox(label="Year released")
            outg15 = gr.Textbox(label="Year released")                             
        with gr.Row():
            outg16 = gr.Textbox(label="Game score")
            outg17 = gr.Textbox(label="Game score")
            outg18 = gr.Textbox(label="Game score")
            outg19 = gr.Textbox(label="Game score")
            outg20 = gr.Textbox(label="Game score")                               
        with gr.Row():
            outg21 = gr.Textbox(label="Game weighted score")
            outg22 = gr.Textbox(label="Game weighted score")
            outg23 = gr.Textbox(label="Game weighted score")
            outg24 = gr.Textbox(label="Game weighted score")
            outg25 = gr.Textbox(label="Game weighted score")                     
        with gr.Row():
            outg26 = gr.Textbox(label="Total ratings")
            outg27 = gr.Textbox(label="Total ratings")
            outg28 = gr.Textbox(label="Total ratings")
            outg29 = gr.Textbox(label="Total ratings")
            outg30 = gr.Textbox(label="Total ratings") 

    btn1.click(fn=greet, inputs = inp1, outputs=outproduct) 
    btn2.click(fn=recommend, inputs = inp2, outputs=[outm1, outm2, outm3, outm4, outm5, outm6, outm7, outm8, outm9, outm10]) 
    btn3.click(fn=recommendb, inputs = inp3, outputs=[out1, out2, out3, out4, out5, out6, out7, out8, out9, out10,out11, out12, out13, out14, out15, out16, out17, out18, out19, out20]) 
    btn4.click(fn=recommendp, inputs = inp4, outputs=[outp1, outp2, outp3, outp4, outp5]) 
    btn5.click(fn=recommendg, inputs = inp5, outputs=[outg1, outg2, outg3, outg4, outg5, outg6, outg7, outg8, outg9, outg10, outg11, outg12, outg13, outg14, outg15, outg16, outg17, outg18, outg19, outg20, outg21, outg22, outg23, outg24, outg25, outg26, outg27, outg28, outg29, outg30]) 

demo.launch(share=True)