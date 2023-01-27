"""

Collaborative filltering :
    
- factorisation de matrice "SVD"
- "KNN" cosine sim

- IMPORTANT : RAM limité (8 Go), donc pour KNN, nrows=10000000

"""

#############################
# Blibliotheques
#############################

import pandas as pd
from collections import defaultdict

from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise import accuracy

from surprise import KNNBasic


#############################
# Chargement
#############################

def read_data() :
    # lecture du dataset anime    
    ratings_df = pd.read_csv('rating.csv')
    anime_df = pd.read_csv('anime.csv')
    
    return ratings_df,anime_df



#############################
# Nettoyage
#############################

def clean_data(anime_df,ratings_df) :
    
    # enlève la ligne avec NaN
    anime_df = anime_df.dropna(axis = 0, how ='any')
    
    
    # on ne prend pas en compte les notes non fournies 
    ratings_df = ratings_df[ratings_df['rating'] != -1]

    # pour eviter que notre mo=atrice ne soit trops creuse
    # on ne prend pas en compte les notes des users que à partir d'un certains nombre de notes
    ratings_df = ratings_df.iloc[:,:].sort_values('user_id')
    counts = ratings_df['user_id'].value_counts()

    ratings_df = ratings_df[ratings_df['user_id'].isin(counts[counts >=  100].index)]
    
    
    return anime_df,ratings_df



#############################
# Format surprise
#############################

def surprise_format_data(ratings_df) :

    reader = Reader(rating_scale=(1, 10))
    # creer la matrice des users et animes
    data = Dataset.load_from_df(ratings_df[['user_id', 'anime_id', 'rating']], reader)    
    
    return data

#############################
# Split
#############################

def split_dataset(data,test_s) :
    
    data_train, data_test = train_test_split(data, test_size=test_s)    
    
    return data_train, data_test
    


#############################
# SVD
#############################

#############################
# KNN
# on choisit d'utiliser car matrice très creuse (beaucoups de 0)
# euclidienne pas approprié
#############################

def init_model(svd=True) :
    
    if (svd) :
        model = SVD(n_factors=100,n_epochs=20)
        return model
    
    # KNN avce cosine sim
    sim_options = {'name': 'cosine', 'user_based': True}
    model = KNNBasic(sim_options = sim_options, verbose = True)    
    return model


#############################
# Apprentissage
#############################

def train_model(model,data_train) :
    
    # apprentissage avec le train set
    model.fit(data_train)
    
    return model



#############################
# Prediction
#############################



def test_model(model,data_test) :

    test_pred = model.test(data_test)
    
    return test_pred


#############################
# Validation des modèles avec RMSE/ MAE
# crash avec KNN, marche avec nrows=100000
#############################

#for i in range (10) :
#    print(i,test_pred[i])
    
# user : id de l'utilisateur
# item : id de l'l'anime
# r_ui : note attribuée par user sur item
# est : estimation 
# {'was_impossible'} : si la prediction est possible ou non    

#     0 user: 67806      item: 9563       r_ui = 8.00   est = 8.30   {'was_impossible': False}
    


def validation(test_pred) : 
    
    print('[INFO]')
    accuracy.rmse(test_pred),accuracy.mae(test_pred)

    #RMSE_OUT : 1.10 (SVD), 1.39 (KNN nrows=100000)
    # avec changement hyper parametre 1.09 (SVD)

    # MAE_OUT : 0.8261 (SVD), 1.0975(KNN nrows=100000)    
    

#############################
# Recommendation & Affichage
#############################


# fonction qui retourne les n meilleurs predictions pour chaque utilisateur du set
def get_top_n(predictions, n=5):

    # dictionnaire
    top_n = defaultdict(list)
    
    # init 
    for user_id, anime_id, rating, prediction, possible in predictions:
         # clé : id de l'utilisateur, 
         # valeur : tuple (id de l'anime, note predit)
        top_n[user_id].append((anime_id, prediction))

    # on parcours le dictionnaire transformée en list avec la methode items()
    for user_id, tuple_anime_id_prediction in top_n.items():
        # on classe les valeurs avec la note predit dans l'ordre decroissant
        tuple_anime_id_prediction.sort(key=lambda x: x[1], reverse=True)
        # on retourne les n meilleurs predictions
        top_n[user_id] = tuple_anime_id_prediction[:n]

    return top_n



# affiche les recommendations pour les n premiers utilisateurs du set de test
def print_n_users_recommendations(top_n, anime_df ,n=5):
    
    animename_df = anime_df[['anime_id','name']]

    for uid, user_ratings in top_n.items():
        print(f"Recommendations pour l'utilisateur {uid}:")
        for (iid, _) in user_ratings:
            anime_index = animename_df.index[animename_df['anime_id']==iid]
            anime_name = animename_df.iloc[anime_index]['name'].tolist()
            print(anime_name)
        print("\n")

        if n <= 0:
            break
        n -= 1
        
        
        
        

menu_options = {
    1: 'SVD',
    2: 'KNN',
    3: 'Quitter',
}

def print_menu(affichage_id=False):
    for key in menu_options.keys():
        print (key, '-', menu_options[key] )
     
        
        
def main():
    
    rating_df,anime_df = read_data() 
    
    anime_df, rating_df = clean_data(anime_df,rating_df)
    
    data = surprise_format_data((rating_df))
    
    data_train, data_test = split_dataset(data, 0.2)
    
    # Choix du model par l'utilisateur
    
    #choix_model = input("\n[INPUT] Choisir le model à utiliser pour la recommendation : ")
    print_menu()
    option = ''
    try:
        option = int(input('[INFO] Veuillez choisir le modèle : '))
    except:
        print('[ERROR] Un chiffre est attendu')
    #Check what choice was entered and act accordingly
    if option == 1:
        # svd
        model = init_model()
    elif option == 2:
        # knn
        model = init_model(svd=False)
    elif option == 3:
        print('[INFO] Au revoir')
        exit()
    else:
        print('[ERROR] Un chiffre en 0-3 est attendu')    
        
    # apprentissage
    model = train_model(model, data_train)
    
    # prediction
    test_pred = test_model(model, data_test)
    
    # validation
    validation(test_pred)
    
    recommencer = True
    
    
    # recommendation
    while recommencer:
        
        n = input("[INPUT] Combien de recommendations voulez-vous pour chaque utilisateur du set de test ? ")
        n_first = input("[INPUT] Afficher les recommendations pour les <...> premiers utilisateurs ? ")        
        
        
        try:
            n = int(n)
            n_first = int(n_first)        
        except ValueError:
            print("[ERROR] Veuillez entrer un nombre entier.")
            continue
        
        print("\n[INFO] Les {} recommendations pour les {} premiers utilisateurs du set de test :\n".format(n, n_first))                
        top_n = get_top_n(test_pred, n=n)
        print_n_users_recommendations(top_n, anime_df,n=n_first-1)
        print('\n-------------------------------------------------\n')        
        
        
    



if __name__ == "__main__":
    
    main()        
        
        
        
        
#############################
# Meilleur hyperparamètre 
#############################

"""
k_factors = [25, 50, 75, 100]
n_epochs = [10, 20, 40]

# CV results
test_rmse = []
params = []

# On parcours pour chaque combinaisons
for k in k_factors:
    for n in n_epochs:
        # init model
        algo = SVD(n_factors=k, n_epochs=n, biased=True, lr_all=0.005, reg_all=0, init_mean=0, init_std_dev=0.01, verbose=False)
        # apprentissage
        algo.fit(data_train)
    
        # test
        predictions_test = algo.test(data_test)
        test_rmse.append(accuracy.rmse(predictions_test, verbose = False))
        
        params.append((k,n))
        
        print("k : "+str(k)+", n : "+str(n)+", rmse : "+str(test_rmse[len(test_rmse)-1]))

"""        
