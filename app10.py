import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# # Collaborative recommendation system


st.set_page_config(layout='wide')

@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    movie_data = joblib.load('movie_data.pkl')
    popular_movie = joblib.load('popular_movie.pkl')
    user_similarity = joblib.load('user_based.pkl')
    item_similarity = joblib.load('item_based.pkl')
    user_pivot = joblib.load('pivot_user.pkl')
    item_pivot = joblib.load('pivot_item.pkl')
    genre_database = joblib.load('genre_database.pkl')
    search_text_database = joblib.load('search_text_database.pkl')

    return model, movie_data, popular_movie, user_similarity, item_similarity, user_pivot, item_pivot, genre_database, search_text_database


model, movie_data, high_rated_movie, user_similarity, item_similarity, user_pivot, item_pivot, genre_database, search_text_database = load_models()


#Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []


if 'recommendation_movie' not in st.session_state:
    st.session_state.recommendation_movie = []
    st.session_state.recommending_rate = []

if 'clicked_button' not in st.session_state:
    st.session_state.clicked_button = False

if 'mode' not in st.session_state:
    st.session_state.mode = None

def get_user_based_recommend_movies(user_id, num_of_recommdation):
    recommend_movies = []
    recommend_rate = []

    # movie recommend on user behavior
    #for new users
    if user_id not in user_pivot.index:
        st.success('Your are new user ')
        genre = ['Adventure', 'Animation', 'Comedy', 'Romance', 
            'Drama', 'Action', 'Crime', 'Horror', 'Thriller',
              'Sci-fi', 'Fantasy', 'Documentary']

        with st.form('my_form', ):
            select_genre = st.multiselect('Select your Genre/Mood :', options=genre)
            if st.form_submit_button('show movies'):
                if select_genre:
                    for i in select_genre:
                        movie = movie_data[movie_data['genres'] == i.lower()].sort_values('avg_rating', ascending=False).head(num_of_recommendation)['title'].tolist()
                        rate = movie_data[movie_data['genres'] == i.lower()].sort_values('avg_rating', ascending=False).head(num_of_recommendation)['avg_rating'].tolist()
                        recommend_movies.extend(movie)
                        recommend_rate.extend(rate)

                    # for both genres
                    select_genre = ' '.join(select_genre)
                    emb = model.encode([select_genre])
                    d, i = genre_database.search(emb, k=10)
                    recommend_movies.extend(movie_data['title'].loc[i[0]].tolist())
                    recommend_rate.extend(movie_data['avg_rating'].loc[i[0]].tolist())

                    st.session_state.recommendation_movie = recommend_movies
                    st.session_state.recommending_rate = recommend_rate

                    if st.session_state.recommendation_movie:
                        cols = st.columns(5)  # 5 columns for grid
                        for i in range(len(st.session_state.recommendation_movie)):
                            full_name = st.session_state.recommendation_movie[i]
                            short_name = full_name if len(full_name) <= 30 else f'{full_name[0:22]}...'
                            with cols[i % 5]:
                                st.image('https://via.placeholder/com/500*750?text=No+Image')
                                st.markdown(f'<p class="movie-title"> {short_name} </p>', unsafe_allow_html=True)
                                st.markdown(f'<p class="gray">⭐{round(st.session_state.recommending_rate[i], 1)}/5 </p>', unsafe_allow_html=True)


        st.subheader("High Rated Movies")
        high_rating_movie, high_rating_rate = get_top_rated_movies()
        cols = st.columns(5)
        for i in range(len(high_rating_movie)): 
            full_name = high_rating_movie[i]
            short_name = full_name if len(full_name) <= 30 else f'{full_name[0:22]}...'
            with cols[i % 5]:
                st.image('https://via.placeholder/com/500*750?text=No+Image')
                st.markdown(f'<p class="movie-title"> {short_name} </p>', unsafe_allow_html=True)
                st.markdown(f'<p class="gray">⭐{round(high_rating_rate[i], 1)}/5 </p>', unsafe_allow_html=True)
    


        

    # for regular user
    else:
        recommend_movies = []
        recommend_rate = []
        movie_index = np.where(user_pivot.index == user_id)[0][0]
        similar_user = sorted(list(enumerate(user_similarity[movie_index])), key=lambda x: x[1], reverse=True)[1:num_of_recommendation+1]

        for i in similar_user:
            recommend_movies.append(user_pivot.columns[i[0]])
        
        recommend_rate.extend(movie_data[movie_data['title'].isin(recommend_movies)]['avg_rating'])

        st.subheader('Recommended Movies')
        cols = st.columns(5)  # 4 columns for grid
        for i in range(len(recommend_movies)):
            full_name = recommend_movies[i]
            short_name = full_name if len(full_name) <= 30 else f'{full_name[0:22]}...'
            with cols[i % 5]:
                st.image('https://via.placeholder/com/500*750?text=No+Image')
                st.markdown(f'<p class="movie-title"> {short_name} </p>', unsafe_allow_html=True)
                st.markdown(f'<p class="gray">⭐{round(recommend_rate[i], 1)}/5 </p>', unsafe_allow_html=True)
        

        # similar movie
        similar_movies = []
        similar_movie_rate = []
        for i in recommend_movies:
            movie_index = np.where(item_pivot.index == i)[0][0]
            # 5 high similarity score
            similar_item = sorted(list(enumerate(item_similarity[movie_index])), key=lambda x: x[1], reverse=True)[1:3]

            for i in similar_item:
                similar_movies.append(item_pivot.index[i[0]])
               
            for i in recommend_movies:
                similar_movie_rate.extend(movie_data[movie_data['title'].isin(similar_movies)]['avg_rating'])
        
        st.subheader('Similar Movies')
        cols = st.columns(5)  # 4 columns for grid
        for i in range(len(similar_movies)):
            full_name = similar_movies[i]
            short_name = full_name if len(full_name) <= 30 else f'{full_name[0:22]}...'
            with cols[i % 5]:
                st.image('https://via.placeholder/com/500*750?text=No+Image')
                st.markdown(f'<p class="movie-title"> {short_name} </p>', unsafe_allow_html=True)
                st.markdown(f'<p class="gray">⭐{round(similar_movie_rate[i], 1)}/5 </p>', unsafe_allow_html=True)



        # Add to history
        st.session_state.history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "recommendations": [i for i in recommend_movies],
            'similar-movies' : [i for i in similar_movies]
        })
  


def get_top_rated_movies():
    top_movie = []
    top_rated = []
    for idx in high_rated_movie['title'].index:
        top_movie.append(high_rated_movie['title'][idx])
        top_rated.append(high_rated_movie['avg_rating'][idx])
      
    return top_movie[0:15], top_rated[0:15]


    
def get_moveie_by_search(search_movie):
    search_movie = search_movie.lower()
    searching_movie_list = []
    searching_rate_list = []
    searching_genre_list = []

    search_embedding = model.encode([search_movie])
    d, idx = search_text_database.search(search_embedding, k=num_of_recommendation)
    
    searching_movie_list.extend(movie_data.loc[idx[0]]['title'].tolist())
    searching_rate_list.extend(movie_data.loc[idx[0]]['avg_rating'].tolist())
    searching_genre_list.extend(movie_data.loc[idx[0]]['genres'].tolist())

    st.subheader('Search result')
    cols = st.columns(5)  # 4 columns for grid
    for i in range(len(searching_movie_list)):
        full_name = searching_movie_list[i]
        short_name = full_name if len(full_name) <= 30 else f'{full_name[0:22]}...'
        with cols[i % 5]:
            st.image('https://via.placeholder/com/500*750?text=No+Image')
            st.markdown(f'<p class="movie-title"> {short_name} </p>', unsafe_allow_html=True)
            st.markdown(f'<p class="gray">⭐{round(searching_rate_list[i], 1)}/5 </p>', unsafe_allow_html=True)

    
    st.subheader("High Rated Movies")
    high_rating_movie,  high_rating_rate = get_top_rated_movies()
    cols = st.columns(5)
    for i in range(len(high_rating_movie)):
        full_name = high_rating_movie[i]
        short_name = full_name if len(full_name) <= 30 else f'{full_name[0:22]}...'
        with cols[i % 5]:
            st.image('https://via.placeholder/com/500*750?text=No+Image')
            st.markdown(f'<p class="movie-title"> {short_name} </p>', unsafe_allow_html=True)
            st.markdown(f'<p class="gray">⭐{round(high_rating_rate[i], 1)}/5 </p>', unsafe_allow_html=True)
    



# Header
st.title("Collaborative Filtering Recommendation System")


# Sidebar
search = st.sidebar.text_input('Search', placeholder='Search movies...')
search_button = st.sidebar.button('Search', use_container_width=True)

st.sidebar.header("Filters")
num_of_recommendation = st.sidebar.slider("Number of Recommendations", 1, 15, 10)



st.markdown("""
        <style>
            .movie-title{
            font-size:12px;
            line-height:1.3;
            word-wrap:break-word;
            white-space:normal;
            text-align:left;
            height:5px;
            }
         .gray{
            font-size:13px;
            color:#fafafa99;
            height: 30px;
            }
        </style>
            """, unsafe_allow_html=True)


# Main Content with Tabs
tab1, tab2 = st.tabs(["Recommend", "History"])

with tab1:

    # Personalized Recommendations
    user_id = st.selectbox("", sorted(range(1, 610+1)), placeholder='Select user', index=None)
    
    if st.button("Get Recommendations"):
        st.session_state.mode = 'user_id'
        st.session_state.user = user_id
    #     st.session_state.clicked_button = True
    
    # if st.session_state.clicked_button:
    #     get_user_based_recommend_movies(user_id, num_of_recommendation)
    
    if search_button:
        st.session_state.mode = 'search'
        st.session_state.search = search
    
    if st.session_state.mode == 'user_id':
        get_user_based_recommend_movies(st.session_state.user, num_of_recommendation)
    
    elif st.session_state.mode == 'search':
        get_moveie_by_search(st.session_state.search)


    

with tab2:
    st.header("Recommendation History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.write("No history yet.")

