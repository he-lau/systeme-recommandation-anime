"""

Content-based filltering :
    
- TF-IDF et cosine similarity

"""

#############################
# Blibliotheques
#############################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#############################
# Chargement
#############################

def read_data() :
    
    # lecture du dataset anime
    anime_df = pd.read_csv('anime.csv')
    
    return anime_df


#############################
# Nettoyage
#############################

def clean_data(anime_df) :
    
    # transformer NaN en "" (string vide)
    anime_df['genre'] = anime_df['genre'].fillna('')
    # transformer la chaîne "genre1,genre2,genre3" en ["genre1","genre2","genre3"]
    
    return anime_df


def charge_donnees():
    
    # charge le dataset "anime.csv"
    anime_df = read_data()

    # init TF-IDF
    tfv = TfidfVectorizer()
    
    # nettoyage
    anime_df = clean_data(anime_df)

    genres_str = anime_df['genre'].str.split(',').astype(str)
    
    # document-term matrix (vecteur)
    tfidf_matrix = tfv.fit_transform(genres_str)

    # calcul du cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)

    # indices des noms pour connaître la position dans la matrice
    indices = pd.Series(anime_df.index, index=anime_df['name'])
    
    return anime_df, indices, cosine_sim


#############################
# Recommendation
#############################

def get_recommendation_genres(title,n,cosine_sim, indices, anime_df):
    # indice de title
    idx = indices[title]

    # le cosine score entre l'anime donné et tous les autres
    cosine_sim_scores = list(enumerate(cosine_sim[idx]))
    
    # on ne recommande pas l'anime donné
    cosine_sim_scores.pop(idx)

    # classer ordre decroissant
    cosine_sim_scores = sorted(cosine_sim_scores, key=lambda x: x[1], reverse=True)

    # top n score
    top_n_cosine_sim_scores = cosine_sim_scores[1:n+1]

    # indices des top n
    anime_indices = [i[0] for i in top_n_cosine_sim_scores]

    # retourne dataframe avec le nom et le genre des animes
    return pd.DataFrame({'Nom': anime_df['name'].iloc[anime_indices].values,
                                 'Genre': anime_df['genre'].iloc[anime_indices].values
                                 })



def main():
    anime_df, indices, cosine_sim = charge_donnees()
    
    recommencer = True
    
    while recommencer:
        
        title = input("\n[INPUT] Entrez le nom d'un anime: ")
        
        # Check if the anime name is in the indices
        if title not in indices:
            print("[ERROR] Anime non trouvé dans la liste. Veuillez réessayer.")
            continue
    
        print('\n------------------------------------------------\n')
        print("[INFO] Anime choisi :\n")
        print(anime_df.iloc[indices[title]])
        print('\n')
        
        # Ask the user for the number of recommendations
        n = input("[INPUT] Combien de recommendations voulez-vous ? ")
        try:
            n = int(n)
        except ValueError:
            print("[ERROR] Veuillez entrer un nombre entier.")
            continue
        
        print("\n[INFO] Les {} recommendations pour {} :\n".format(n, title))
        print(get_recommendation_genres(title,n=n,cosine_sim=cosine_sim,indices=indices,anime_df=anime_df))
        print('\n-------------------------------------------------\n')




if __name__ == "__main__":
    main()
















