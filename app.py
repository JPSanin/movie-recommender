import dash
from dash import dcc
from dash import html
from dash import ctx
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans


app = dash.Dash(__name__)


total_movies = 0
user_movies = ["1. Empty \n", "2. Empty \n",
                "3. Empty \n", "4. Empty \n", "5. Empty \n"]
click_counter = 0
user_ratings = pd.DataFrame(columns=['userId','movieId','rating'])
user_movs = pd.DataFrame(columns=['movieId','original_title','genres'])

# Model Development and Implementation
movies_df = pd.read_csv('movies_metadata.csv')
ratings_df = pd.read_csv('ratings.csv')

movies_df['genres'] = movies_df['genres'].str.strip('[]').str.replace('{', '').str.replace('}', '').str.replace(
    'id', '').str.replace('name', '').str.replace("'", '').str.replace(':', '').str.replace(",", "").str.replace(" ", "")
movies_df['genres'].replace(r'[0-9]', '', regex=True, inplace=True)
movies_df['genres'].replace(
    r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', regex=True, inplace=True)
movies_df['genres'].replace(' ', ',', regex=True, inplace=True)

movies_df_en = movies_df[movies_df['original_language'] == "en"]
movies_df_filtered = movies_df_en[movies_df['genres'] != ""]
movies = movies_df_filtered[['id', 'original_title', 'genres']]
movies = movies.rename(columns={"id": "movieId"})
movies['movieId'] = movies['movieId'].astype(str).astype(int)


def get_genre_ratings(ratings, movies, genres, column_names):
    genre_ratings = pd.DataFrame()
    for genre in genres:
        genre_movies = movies[movies['genres'].str.contains(genre)]
        avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, [
            'userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)

        genre_ratings = pd.concat(
            [genre_ratings, avg_genre_votes_per_user], axis=1)

    genre_ratings.columns = column_names
    return genre_ratings


genre_ratings_3 = get_genre_ratings(ratings_df, movies,
                                    ['Comedy', 'Romance', 'Action'],
                                    ['avg_comedy_rating', 'avg_romance_rating', 'avg_action_rating'])
genre_ratings_3 = genre_ratings_3.dropna()

kmin = 1
kmax = 10
init = 'k-means++'
n_init = 10
max_iter = 300
random_seed = 42

def model(df_x_train, k, kmin, kmax, init, n_init, max_iter, random_seed):
    kmeans = KMeans(n_clusters=k,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_seed)
    kmeans.fit(df_x_train)
    return kmeans

kmeans = model(genre_ratings_3, 6, kmin, kmax, init, n_init, max_iter, random_seed)
centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ['Comedy', 'Romance', 'Action'])
kmeans.cluster_centers_[0] = (kmeans.cluster_centers_[0]+kmeans.cluster_centers_[4])/2
kmeans.cluster_centers_[2] = (kmeans.cluster_centers_[2]+kmeans.cluster_centers_[3])/2
centroids = pd.DataFrame(kmeans.cluster_centers_, columns = ['Comedy', 'Romance', 'Action'])
update_centroids = centroids.drop(3)
update_centroids = update_centroids.drop(4)
kmeans.cluster_centers_ = update_centroids.to_numpy()
kmeans.labels_[kmeans.labels_ == 4] = 0
kmeans.labels_[kmeans.labels_ == 3] = 2


app.layout = html.Div(children=[

    dcc.ConfirmDialog(
        id='movie-genre-error',
        message='Movie found but not in our genres. Please add a movie of the genres Comedy, Romance or Action',
    ),

    dcc.ConfirmDialog(
        id='movie-notfound-error',
        message='The Movie you entered was not found in our database, we are sorry please try another one',
    ),

    dcc.ConfirmDialog(
        id='no-group-error',
        message='You were not placed in any group, we are sorry about that, maybe our genres arent for you',
    ),

    html.Div(id="header", children=[
        html.H1(children='TheMovieClub.com')
    ]),


    html.Div(id="section1", children=[

        html.H3(children='Our movie club consists of three groups based on the movie genres: Action, Comedy, and Romance'),

        html.H2(id="our-groups", children='Our groups '),
        html.P(id="members", children='Current Total members: 125,767'),

        html.Div(id="section2", children=[
            html.Div(className="card", children=[
                html.H2(className="card-title", children='Action Packed'),
                html.P(className="card-subtitle",
                        children='Total members: 12,840'),
            ]),

            html.Div(className="card", children=[
                html.H2(className="card-title", children='Rom-Com'),
                html.P(className="card-subtitle",
                        children='Total members: 11,782'),
            ]),

            html.Div(className="card", children=[
                html.H2(className="card-title", children='All around'),
                html.P(className="card-subtitle",
                        children='Total members: 101,145'),
            ])
        ]),


        html.Div(id="predictSection", children=[

            html.Div(children=[
                html.H3(children='Please add 5 movies to get you started'),
            ]),

            html.Div(id="inputs", children=[
                html.Div(children=[
                    html.Label('Movie Title: '),
                    dcc.Input(id='movie-title', value='',
                                placeholder='Enter a movie title', type='text'),
                ]),

                html.Div(children=[
                    html.Label('Rating (1-5): '),
                    dcc.Input(
                        id='rating', value='', placeholder='Enter rating', min=1, max=5, type='number'),
                ]),
            ]),

            html.Button('Add Movie', id='submit-button', n_clicks=0),

            html.Div(id='user-movies', children='Please enter a movie',
                        style={'white-space': 'pre'}),

            html.Div(id="buttons", children=[
                html.Button('Clear Movies', id='clear-button', n_clicks=0),
                html.Button('Place Me!', id='proceed-button', n_clicks=0),
            ]),

        ]),

        
            html.Div(children=[
            html.H3(children='Your results'),
        ]),

        html.Div(id="results", children='Submit all 5 movies to view',
                        style={'white-space': 'pre'}),

       

    ]),

])


@app.callback(
    Output('user-movies', 'children'),
    Output('movie-title', 'value'),
    Output('rating', 'value'),
    Output('movie-genre-error', 'displayed'),
    Output('movie-notfound-error', 'displayed'),
    Output('no-group-error', 'displayed'),
    Output('results', 'children'),
    Input('movie-title', 'value'),
    Input('rating', 'value'),
    Input('submit-button', 'n_clicks'),
    Input('clear-button', 'n_clicks'),
    Input('proceed-button', 'n_clicks'),
)
def update_output(movie, rating, n_clicks__s, nclicks_c, nclicks_p):
    global total_movies
    global user_movies
    global user_ratings
    global user_movs
    if ("submit-button" == ctx.triggered_id and total_movies < 5):
        i=movies[movies["original_title"]==movie]
        if("Comedy" in str(i["genres"]) or "Action" in str(i["genres"]) or "Romance" in str(i["genres"])):
            user_movs.loc[len(user_movs.index)] = i.values[0]
            user_ratings.loc[len(user_ratings.index)] = [1, i.values[0][0], rating]
            if (total_movies == 0):
                user_movies[0] = '1. Movie title: {}, Rating: {}\n'.format(
                    movie, rating)
            elif (total_movies == 1):
                user_movies[1] = '2. Movie title: {}, Rating: {}\n'.format(
                    movie, rating)
            elif (total_movies == 2):
                user_movies[2] = '3. Movie title: {}, Rating: {}\n'.format(
                    movie, rating)
            elif (total_movies == 3):
                user_movies[3] = '4. Movie title: {}, Rating: {}\n'.format(
                    movie, rating)
            elif (total_movies == 4):
                user_movies[4] = '5. Movie title: {}, Rating: {}\n'.format(
                    movie, rating)
            total_movies += 1
            return user_movies[0]+user_movies[1]+user_movies[2]+user_movies[3]+user_movies[4], '', '', False, False, False, 'Submit all 5 movies to view'
        elif(i.empty == False):
            return user_movies[0]+user_movies[1]+user_movies[2]+user_movies[3]+user_movies[4], '', '', True, False, False, 'Submit all 5 movies to view'
        else:
            return user_movies[0]+user_movies[1]+user_movies[2]+user_movies[3]+user_movies[4], '', '', False, True, False, 'Submit all 5 movies to view'

    if ("clear-button" == ctx.triggered_id):
        total_movies = 0
        user_movies = ["1. Empty \n", "2. Empty \n",
                        "3. Empty \n", "4. Empty \n", "5. Empty \n"]
        user_ratings = pd.DataFrame(columns=['userId','movieId','rating'])
        user_movs = pd.DataFrame(columns=['movieId','original_title','genres'])
        return user_movies[0]+user_movies[1]+user_movies[2]+user_movies[3]+user_movies[4], '', '', False, False, False, 'Submit all 5 movies to view'

    if ("proceed-button" == ctx.triggered_id):
        if(total_movies==5):
            user_final_ratings = get_genre_ratings(user_ratings, user_movs, [ 'Comedy', 'Romance', 'Action'], [ 'avg_comedy_rating', 'avg_romance_rating','avg_action_rating'])
            user_final_ratings = user_final_ratings.fillna(2.5)
            x = kmeans.predict(user_final_ratings)
            x1 = kmeans.transform(user_final_ratings)

            if(x[0]==0):
                return user_movies[0]+user_movies[1]+user_movies[2]+user_movies[3]+user_movies[4], '', '', False, False, True, 'No group matches your taste :('
            else:
                x1 = np.delete(x1, 0)
                x1=sorted(range(len(x1)), key=lambda k: x1[k])
                x1 = np.array(x1)
                x1 = x1.astype('U256')
                x1[x1=='0'] = "Rom-Com"
                x1[x1=='1'] = "All-Around"
                x1[x1=='2'] = "Action Packed"
                out = 'Group Afinity Ranking: \n 1.  {} \n 2.  {} \n 3.  {} \n'.format(
                    x1[0], x1[1], x1[2])
                return user_movies[0]+user_movies[1]+user_movies[2]+user_movies[3]+user_movies[4], '', '', False, False, False,  out
        else:
            return user_movies[0]+user_movies[1]+user_movies[2]+user_movies[3]+user_movies[4], '', '', False, False, False,  'Submit all 5 movies to view'

if __name__ == '__main__':
    app.run_server(debug=True)
