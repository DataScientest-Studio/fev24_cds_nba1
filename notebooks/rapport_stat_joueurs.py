import pandas as pd
import numpy as np

#import des fichiers csv
df = pd.read_csv('C:/Users/mboko/PycharmProjects/nba_intro/Players.csv')

df1 = pd.read_csv('C:/Users/mboko/PycharmProjects/nba_intro/Seasons_Stats.csv')

df2 = pd.read_csv('C:/Users/mboko/PycharmProjects/nba_intro/seasons_stats_2018-2020.csv')

df3 = pd.read_csv('C:/Users/mboko/PycharmProjects/nba_intro/player_data.csv')

#ajout des données de df2 à la fin de df1
df1 = pd.concat([df1, df2])

#création des stat des joueurs par match
df1['PTM'] = df1['PTS'] / df1['G']
df1['ASTM'] = df1['AST'] / df1['G']
df1['STLM'] = df1['STL'] / df1['G']
df1['BLKM'] = df1['BLK'] / df1['G']
df1['TOVM'] = df1['TOV'] / df1['G']
df1['TRBM'] = df1['TRB'] / df1['G']

#création du dataframe df1_annexe comportant les données de df1 que l'on souhaite garder
df1_annexe = df1[['Player','Age', 'Year', "Pos",'TS%','PTM','ASTM','TRBM','STLM','BLKM','TOVM','USG%','FG%','2P%','3P%','FT%','PTS']]

# Séparer les colonnes numériques des colonnes non numériques
numeric_columns = df1_annexe.select_dtypes(include=[np.number]).columns
non_numeric_columns = df1_annexe.select_dtypes(exclude=[np.number]).columns

# Calculer la moyenne des colonnes numériques pour les années en double ou triple
df_grouped = df1_annexe.groupby(['Player', 'Year'], as_index=False)[numeric_columns].mean()

# Conserver les premières valeurs des colonnes non numériques
df_non_numeric = df1_annexe.groupby(['Player', 'Year'], as_index=False)[non_numeric_columns].first()

# Fusionner les DataFrames pour obtenir le DataFrame final
df_final = pd.merge(df_grouped, df_non_numeric, on=['Player', 'Year'])

print("\nColonnes du DataFrame final après fusion:")
print(df_final.columns)

# Vérifier les valeurs manquantes
missing_values = df1_annexe[['Player', 'Year']].isna().any()
print("\nValeurs manquantes dans df1_annexe:")
print(missing_values)

# Supprimer les lignes avec des valeurs manquantes dans 'Player' ou 'Year'
df1_annexe_clean = df1_annexe.dropna(subset=['Player', 'Year'])

# Vérifier les valeurs manquantes après nettoyage
missing_values_clean = df1_annexe_clean[['Player', 'Year']].isna().any()
print("\nValeurs manquantes après nettoyage:")
print(missing_values_clean)

# Réordonner le DataFrame selon l'ordre original des années
df1_grp = df_final.set_index(['Player', 'Year']).loc[df1_annexe_clean.set_index(['Player', 'Year']).index.drop_duplicates()].reset_index()

# on restreint le nombre de joueurs aux joueurs qui nous intéresse
df1_grp1 = df1_grp.loc[(df1_grp['Player'] == 'LeBron James') | (df1_grp['Player'] == 'Stephen Curry') | (df1_grp['Player'] == 'Giannis Antetokounmpo') | (df1_grp['Player'] == 'Kobe Bryant') | (df1_grp['Player'] == 'Kevin Durant') | (df1_grp['Player'] == 'Dwyane Wade') | (df1_grp['Player'] == 'Dirk Nowitzki') | (df1_grp['Player'] == 'Tim Duncan') | (df1_grp['Player'] == "Shaquille O'Neal*") | (df1_grp['Player'] == 'Steve Nash') | (df1_grp['Player'] == 'Kawhi Leonard') | (df1_grp['Player'] == 'James Harden') | (df1_grp['Player'] == 'Jason Kidd') | (df1_grp['Player'] == 'Allen Iverson*') | (df1_grp['Player'] == 'Chris Webber') | (df1_grp['Player'] == 'Kevin Garnett') | (df1_grp['Player'] == 'Paul Pierce') | (df1_grp['Player'] == 'Jimmy Butler') | (df1_grp['Player'] == 'Russell Westbrook') | (df1_grp['Player'] == 'Dwight Howard') ]

# Remplacer le caractère '*' à la fin des noms
df1_grp1['Player'] = df1_grp1['Player'].str.replace(r'\*$', '', regex=True)

#On tri le dataframe df3 par nom
df3_fil = df3.sort_values(by = 'name')
#on remplace le nom de la colonne name par Player en vue d'une fusion
df3_fil1 = df3_fil.rename(columns={"name": "Player"}, inplace=True)

#fusion de df1_grp1 et df3_fil
df_fin = df1_grp1.merge(df3_fil, how = "left", left_on = "Player", right_on = "Player" )

#on ne garde à présent que ce qui se passe à partir de l'an 2000
df_rap = df_fin.loc[df_fin['Year'] < 2000].index
df_fin = df_fin.drop(df_rap)

# Trier le DataFrame par la colonne 'Player' dans l'ordre alphabétique
df_fin = df_fin.sort_values(by=['Player','Year'])

#on ne garde que les variables utile pour la réussite au shoot
df_rapport = df_fin.drop(['STLM','BLKM','position','birth_date','college','ASTM','TOVM','college'], axis = 1)
df_rapport['height'] = df_rapport['height'].str.replace('-', '.')

# Convertir la colonne 'height' en float
df_rapport['height'] = df_rapport['height'].astype(float)
#taille mis en cm
df_rapport['height'] = df_rapport['height'] * 30.48

dummies_data = pd.get_dummies(df_rapport['Pos'])
# Convertir les booléens en entiers
df_dummies = dummies_data.astype(int)

#suppression de la colonne Pos
df_rapport = df_rapport.drop(['Pos'], axis = 1)
#fusion des dataframe df_rapport et df_dummies
df_rapport = pd.concat([df_rapport, df_dummies], axis=1)

print(df_rapport.head(14))

