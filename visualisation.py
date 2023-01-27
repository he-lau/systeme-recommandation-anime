"""

Visualisation du dataset  :

- anime.csv
- rating.csv    

"""

#############################
# Blibliotheques
#############################

import pandas as pd
import matplotlib.pyplot as plt

#############################
# Chargement 
#############################

# lecture du dataset anime
rating_df = pd.read_csv('rating.csv')
#ratings_df = pd.read_csv('rating.csv',nrows=100000)
anime_df = pd.read_csv('anime.csv')


#############################
# Nettoyage & visualisation
#############################

anime_df.head()
rating_df.head()

print(anime_df.shape) # (12294, 7)
print(rating_df.shape)# (7813737, 3)

print(anime_df.isnull().sum())
print(rating_df.isnull().sum())

# enlève la ligne avec NaN
anime_df= anime_df.dropna(axis = 0, how ='any')

print(anime_df['name'].unique()[:10])

"""
['Kimi no Na wa.' 'Fullmetal Alchemist: Brotherhood' 'Gintama°'
 'Steins;Gate' 'Gintama&#039;'
 'Haikyuu!!: Karasuno Koukou VS Shiratorizawa Gakuen Koukou'
 'Hunter x Hunter (2011)' 'Ginga Eiyuu Densetsu'
 'Gintama Movie: Kanketsu-hen - Yorozuya yo Eien Nare'
 'Gintama&#039;: Enchousen']
"""

print(anime_df['genre'].unique()[:10])

# 2d --> 1d
def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


plt.rcParams['figure.figsize'] = [6, 8]

genre_list = list()

for i in range(len(anime_df['anime_id'])) :
    genre_list.append([x.strip() for x in str(anime_df.iloc[:,:-1].values[i][2]).split(',')])

plt.title('Anime genre repartition')
plt.hist(flatten_list(genre_list), orientation='horizontal')
plt.ylabel('')
plt.grid()
plt.figure()
plt.show()

print(anime_df['type'].value_counts())
plt.title('Anime type repartition')
plt.hist(anime_df['type'])
plt.grid()
plt.show()

# beaucoups de -1 (pas de note) #1476496
print(rating_df['rating'].value_counts())
rating_df = rating_df[rating_df['rating'] != -1]


plt.title('Notation repartition')
plt.hist(rating_df['rating'])
plt.grid()
plt.show()




    
