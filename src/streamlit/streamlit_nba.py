#importer les packages
import pandas as pd
import streamlit as st
import os
import matplotlib.pyplot as plt
from src.models.predict_model import predict
#import seaborn as sns

#import pickle
#from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    df_fin = pd.read_csv('data/processed/stat_joueurs_streamlit.csv')
    df_pbp_sample = pd.read_csv('data/raw/missing_pbp_2019-2020.csv', nrows=5, index_col=0)
    df_all_shots = pd.read_csv('data/processed/all_shots-v6.csv', index_col=0)
    return df_fin, df_pbp_sample, df_all_shots

# Agrandir la colonne principale d'affichage
st.set_page_config(layout="wide")

# Chargement des données
df_fin, df_pbp_sample, df_all_shots = load_data()

# Sidebar de navigation
st.sidebar.title("Sommaire")
pages = ["Description du projet", "Exploration des données", "Stat des  joueurs",  "Preprocessing",  "Modélisation", "Démo !", "Conclusion"]
page = st.sidebar.radio("Aller vers la page :", pages)



    # # Affichez le bouton de choix de
    # if st.sidebar.button('Choisir la page'):
    #     st.sidebar.write("Bouton de choix de page cliqué")
    #     page = pages[3]


#########################################################################################################################################
#                                                           PAGE DESCRIPTION DU PROJET                                                  #
#########################################################################################################################################

if page == pages[0] :
    st.write('# Description du projet')
    st.write("Les sports américains sont très friands de statistiques, et la NBA (National Basketball Association) ne fait pas exception à la règle. ")

    st.write("Le développement constant des nouvelles technologies et des outils numériques, permet désormais de suivre en temps réel les déplacements de tous les joueurs sur un terrain de basket. Les données recueillies sont ainsi très nombreuses et riches.")

    st.write("Le but de ce projet est de : ")

    st.write("    1.  Comparer les tirs (fréquence et efficacité au tir par situation de jeu et par localisation sur le terrain) de 20 des meilleurs joueurs de NBA du 21ème siècle.")

    st.write("    2.  Pour chacun de ces 20 joueurs encore actifs aujourd’hui (de LeBron James à Giannis Antetokounmpo), estimer à l’aide d’un modèle la probabilité qu’a leur tir de rentrer dans le panier, en fonction de différentes métriques.")

    st.write("## Problématique")

    st.write("Nous sommes face à un problème de classification, nous devons déterminer si le tir rentre ou non dans le panier en nous basant sur les données de localisation du tir, les performances du joueur et de son équipe, la situation de jeu. ")

    st.write("## Datasets")
    """
    - dataset des tirs NBA entre 1997 et 2019 : https://www.kaggle.com/jonathangmwl/nba-shot-locations
- dataset des actions de chaque match entre 2000 et 2020 : https://sports-statistics.com/sports-data/nba-basketball-datasets-csv-files/
- dataset des bilans d’équipe entre 2014 et 2018 : https://www.kaggle.com/nathanlauga/nba-games?select=ranking.csv
- dataset des joueurs de NBA depuis 1950 : https://www.kaggle.com/drgilermo/nba-players-stats?select=Players.csv

    """
    st.write("L’ensemble de ces données peuvent également être récupérées via des API qui scrappent NBA.com, le site officiel.")

    st.write("Notamment https://github.com/swar/nba_api, grâce aux contributeurs que nous pourrons utiliser afin d’obtenir les données actualisées depuis le site NBA.com.")

    st.write("L’API fournit de nombreuses méthodes permettant de récupérer les joueurs et leur stats, l’historique des matches et le play by play pour chacun d’eux.")

    st.write("Sur ce bonne exploration")

    st.image("reports/pictures/jordan.jpg")


#########################################################################################################################################
#                                                           PAGE EXPLORATION DES DONNEES                                                #
#########################################################################################################################################


elif page == pages[1]:
    st.write("# Exploration des données")

    """Pour ce projet, les données viennent de multiples sources.
    Les datasets à notre disposition sont les actions de jeu (play by play), les localisations des tirs (shot locations) ainsi que les statistiques par joueur et par équipe pour chaque saison.
    """

    st.write("## Play by Play")
    """Les fichiers 'play by play' (ou action par action en français) recensent pour chaque année entre 2000 et 2020 l’intégralité des actions de jeu, de l’entre-deux jusqu’au buzzer final. Ils contiennent 33 colonnes et chacun entre 500k et 600k lignes.
    """

    st.dataframe(df_pbp_sample)

    st.write("### Analyse univariée")

    "Comme on peut le voir ci-dessus, les play by play contiennent de nombreuses **valeurs manquantes** et l'encodage des variables n'est pas forcément intelligible. Ci-dessous l'ensemble des types d'actions:"
    "Pour éviter une fuite de données nous supprimerons les colonnes relatives au player2 et player3 qui sont respectivement présentes lorsqu'il y a une passe décisive (=tir marqué) ou un block (=tir raté) et ne peuvent pas être connue à priori."

    _, cent_co, _ = st.columns([1,3,1])
    with cent_co:
        st.image("src/streamlit/figures/action_types.png")


    "Après exploration du fichier nous décidons de garder les actions dont la variable EVENTMSGTYPE est :"
    st.markdown("""
                * 1 : Tir marqué
        * 2 : Tir raté
        * 3 : Lancer franc
    """)
    "Une fois les tirs identifiés et filtrés, nous pouvons regarder la distribution de notre **variable cible**. Les classes sont légèrement déséquilibrées avec plus de tir réussis que ratés."
    _, cent_co, _ = st.columns([1,3,1])
    with cent_co:
        st.image(["src/streamlit/figures/class_distribution.png"])



    st.write("### Analyse bivariée")
    "Dans cette section nous cherchons à analyser l'impact d'une variable sur la variable cible."

    col1, col2 = st.columns(2)
    with col1:
        st.image("src/streamlit/figures/shots_per_player.png")
    with col2:
        st.image("src/streamlit/figures/shots_per_team.png")

    "Nous pouvons voir que les joueurs et les équipes ne tirent pas le même nombre de tirs. Les joueurs les plus prolifiques sont souvent les plus talentueux et les équipes les plus performantes sont celles qui tirent le plus."
    "Notre dataset est donc **déséquilibré** en terme de nombre de tirs par joueur et par équipe."


    _, cent_co, _ = st.columns([1,3,1])
    with cent_co:
        st.image(["src/streamlit/figures/success_by_shot.png"])
        st.image(["src/streamlit/figures/success_by_previous_actions.png"])

    """Nous observons que le taux de réussite varie en fonction du type de tir et de l'action précédente. A titre d'exemple, les layups sont plus souvent mis que ratés et inversement pour les jump shots.
    Par ailleurs, un shot pris après un rebond offensif est plus souvent réussi qu'après un rebond défensif."""


    st.write("## Shots locations")

        ##### A REMPLIR FATIHA #####
    """Le fichier 'Shot locations' recensent les positions/distance/zone de tirs dans le terrain pour chaque joueur pour les années entre 1997 et 2020. Il contient 22 colonnes avec 4.7 millions lignes. Après la selection des 20 meilleurs joueurs entre 2000 et 2020, on obtient autour de 300K entrées sans doublons ou valeurs manquantes.
    """

    df_shotLocations = pd.read_csv('data/processed/Shot_Locations_top_20_players_2000to2020.csv')
    st.dataframe(df_shotLocations)

    st.image("src/streamlit/figures/distributions_tirs_XYLocations_shotZoneBasic_Top20Players.png")

    st.image("src/streamlit/figures/Capture_tirs_shot_zone_basic_ShotMadeFlag.png")

    st.image("src/streamlit/figures/distributions_tirs_ShotDistance_NotNormalized_ShotMadeFlag_KDE_AllPlayers.png")

    st.image("src/streamlit/figures/distributions_tirs_MinutesRemaining_NotNormalized_ShotMadeFlag_AllPlayers.png")

    st.image("src/streamlit/figures/distributions_tirs_SecondsRemaining_NotNormalized_ShotMadeFlag_AllPlayers.png")
    


#########################################################################################################################################
#                                                           PAGE STATS JOUEURS                                                          #
#########################################################################################################################################


elif page == pages[2]:

    st.write("# Stats des joueurs")

    st.write("Ici nous pouvons voir les statistiques des joueurs ainsi que leurs positions préférentielles sur le terrain")
    blank = ['Choisir un joueur']
    liste_joueurs = df_fin['Player'].unique()
    blank.extend(liste_joueurs)

    # Add a selectbox to the sidebar:
    result_joueur = st.selectbox(
        'Choisir un joueur', blank
    )

    if result_joueur != '':
        # Récupérer les saisons disponibles pour le joueur sélectionné
        annees_disponibles = df_fin['Year'].loc[df_fin['Player'] == result_joueur].unique()

        # Sidebar pour sélectionner la saison
        result_annee = st.selectbox("Choisir une saison:", annees_disponibles)

        # Afficher la saison sélectionnée
        st.write(f"La saison sélectionnée est: {result_annee}")

    st.write("###", result_joueur)
    if not result_joueur == "Choisir un joueur":
        pts_moyen = df_fin['PTM'].loc[(df_fin['Player'] == result_joueur) & (df_fin['Year'] == result_annee)].unique()[0]

    # Création des colonnes
    col1, col2 = st.columns(2)

    # Contenu de la colonne de gauche
    with col1:
        st.write("Moyenne de points marqués par match sur la saison")

    df_all_shots.drop(df_all_shots.loc[df_all_shots['free_throw'] == 1].index, inplace=True)
    df_all_shots['X Location'] = df_all_shots['X Location'] / 10
    df_all_shots['Y Location'] = df_all_shots['Y Location'] / 10

    # Contenu de la colonne de gauche
    def get_image_path(player_name):
        return f"reports/pictures/{result_joueur}.png"


    # Obtenir le chemin de l'image
    image_path = get_image_path(result_joueur)

    with col2:
        # Vérifier si le fichier existe
        if os.path.exists(image_path):
            st.image(image_path, width=250)
        else:
            st.write(f"L'image pour {result_joueur} n'a pas été trouvée à l'emplacement {image_path}.")
        #st.image(image_path)


    # Définition de la fonction de transformation
    def custom_round(value):
        base = (value // 3) * 3  # Trouver la base (0, 3, 6, etc.)
        if value % 3 < 1.5:
            return base + 1.5  # Pour les valeurs de 0 à 2.9, ajuster au centre de l'intervalle inférieur
        else:
            return base + 1.5  # Pour les valeurs de 3.0 à 5.9, ajuster au centre de l'intervalle supérieur


    # Application de la fonction aux colonnes X et Y
    df_all_shots['X Location'] = df_all_shots['X Location'].apply(custom_round)
    df_all_shots['Y Location'] = df_all_shots['Y Location'].apply(custom_round)

    df_n = df_all_shots[['PLAYER1_NAME', 'Year', 'X Location', 'Y Location', 'target']]

    # Fonction pour convertir l'année en format 2003-04
    def convert_year_format(year):
        start_year = year - 1
        end_year = str(year)[-2:]
        return f"{start_year}-{end_year}"

    # Convertir la colonne Year en chaîne de caractères
    df_n['Year'] = df_n['Year'].astype(str)

    # Appliquer la fonction à la colonne Year
    df_n['Year'] = df_n['Year'].apply(lambda x: f"{int(float(x)) - 1}-{str(int(float(x)))[-2:]}")

    # Combiner les colonnes X et Y pour obtenir la position
    df_n['Position'] = df_n[['X Location', 'Y Location']].apply(tuple, axis=1)

    # Grouper par joueur, année et position, et ajouter la somme de la colonne 'target' ainsi que le nombre de positions
    position_counts = df_n.groupby(['PLAYER1_NAME', 'Year', 'Position']).agg({
        'target': ['sum', 'count']
    }).reset_index()

    # Renommer les colonnes pour une meilleure lisibilité
    position_counts.columns = ['PLAYER1_NAME', 'Year', 'Position', 'Total_Target', 'Count']

    # Trier par joueur, année et nombre de positions, puis sélectionner les 10 premières positions par groupe
    top_positions = position_counts.sort_values(['PLAYER1_NAME', 'Year', 'Count'], ascending=[True, True, False])
    top_positions = top_positions.groupby(['PLAYER1_NAME', 'Year']).head(10)

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg


    def draw_basketball_court(joueur, annee):
        fig, ax = plt.subplots(figsize=(8, 8))
        background_img = mpimg.imread('reports/figures/saved_court.png')

        # Afficher l'image de fond
        ax.imshow(background_img, extent=[0, 50, 0, 50], aspect='auto')

        # Dimensions du terrain de basket-ball en pieds
        court_length_ft = 47
        court_width_ft = 50


        # Extraire les coordonnées X et Y des tuples dans la colonne 'Position'
        top_positions['X'] = top_positions['Position'].apply(lambda pos: pos[0])
        top_positions['Y'] = top_positions['Position'].apply(lambda pos: pos[1])

        # Filtrer par joueur et année si nécessaire
        player = joueur  # Exemple de joueur
        year = annee
        # Exemple d'année
        filtered_positions = top_positions[(top_positions['PLAYER1_NAME'] == player) & (top_positions['Year'] == year)]

        # Déplacer l'origine des axes à (25, 5.5)
        origin_x, origin_y = 25, 5.5

        # Ajuster les limites des axes en fonction de la nouvelle origine
        ax.set_xlim(left=origin_x - 50, right=origin_x + 50)  # Ajuster selon vos besoins
        ax.set_ylim(bottom=origin_y - 50, top=origin_y + 50)  # Ajuster selon vos besoins

        # Ajuster les coordonnées des données en fonction de la nouvelle origine
        adjusted_x = filtered_positions['X'] + origin_x
        adjusted_y = filtered_positions['Y'] + origin_y

        # Tracer les points
        ax.scatter(adjusted_x, adjusted_y, s=filtered_positions['Count'] * 10, alpha=0.5)

        # Configurations des axes
        ax.set_xlim(0, court_width_ft)
        ax.set_ylim(0, court_length_ft)
        ax.set_aspect('equal')
        ax.set_title('Terrain de basket-ball NBA')
        ax.set_xlabel('Largeur (pieds)')
        ax.set_ylabel('Longueur (pieds)')

        st.pyplot(fig)

    # Filtrer les données pour un joueur
    df_player = df_fin.loc[df_fin['Player'] == result_joueur]
    df_player['PTM'] = df_player['PTM'].round(1)

    # Créer le diagramme en barres horizontales
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.barh(df_player['Year'], df_player['PTM'], color='skyblue')

    # Enlever toutes les spines
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # Enlever les ticks et les labels de l'axe x
    ax2.xaxis.set_ticks([])
    ax2.xaxis.set_ticklabels([])

    # Conserver les labels de l'axe y
    ax2.yaxis.set_ticks_position('none')  # Enlever les ticks tout en gardant les labels

    # Ajouter les valeurs au bout des barres et mettre en gras la valeur pour l'annee '2018-19'
    for i, bar in enumerate(bars):
        width = bar.get_width()  # Largeur de la barre
        y_pos = bar.get_y() + bar.get_height() / 2  # Position centrale verticale

        # Trouver l'année associée à la barre
        year = df_player.iloc[i]['Year']

        # Vérifier si l'année est result_annee et définir le poids du texte
        fontweight = 'bold' if year == result_annee else 'normal'

        # Ajouter le texte au bout de la barre avec un petit décalage pour la lisibilité
        ax2.text(
            width + 0.3,  # Décalage de 1 unité à droite de la barre pour la lisibilité
            y_pos,
            f'{width}',
            va='center',  # Alignement vertical au centre
            ha='left',  # Alignement horizontal à gauche pour placer le texte à droite de la barre
            fontsize=12,
            weight=fontweight
        )

    # Contenu de la colonne de gauche
    with col1:
        # Afficher le graphique
        st.pyplot(fig2)

    # Appel de la fonction pour dessiner le terrain de basket-ball
    draw_basketball_court(result_joueur, result_annee)



#########################################################################################################################################
#                                                           PAGE PREPROCESSING                                                         #
#########################################################################################################################################
elif page == pages[3]:
    st.write("# Preprocessing")
    """L'exploration des données nous a permi de sélectionner et nettoyer les variables nécessaires à la modélisation.
    Ensuite nous avons fusionné les différents datasets pour n'en créé qu'un:"""

    st.image("src/streamlit/figures/data_merge.png")

    """Notre dataset est désormais nettoyé et fussioné, il ne contient plus de valeurs manquantes ni doublons et il ne contient que les actions de tir des 20 meilleurs joueurs du 21ème siècle."""


    st.dataframe(df_all_shots.head(10))

    """Cependant, il contient encore des variables non numériques comme les abbréviations des équipes, ainsi que des variables numériques qui risquent de fausser les résultats du modèles comme le numéro du match.
    Et surtout, notre dataset fusionné contient beaucoup de variables ce qui risque de poser divers problèmes :"""
    st.markdown("""
        * Risque de surapprentissage
* Temprs d'entrainement accru
* Multi colinéarité entre les variables
    """)

    st.write("#### Étapes de Preprocessing")

    st.write("Une fois que le dataset est nettoyé, les étapes de preprocessing suivantes sont nécessaires :")

    st.markdown("""
    1. **Encodage des variables catégorielles** :
        - Convertir les variables catégorielles en variables numériques à l'aide de techniques comme le One-Hot Encoding.
    2. **Normalisation des données** :
        - Appliquer une normalisation ou une standardisation aux variables numériques pour s'assurer qu'elles ont une échelle similaire.
    3. **Séparation des données** :
        - Diviser le dataset en ensembles d'entraînement et de test pour évaluer les performances du modèle.
    4. **Feature Engineering** :
        - Créer de nouvelles variables à partir des données existantes pour améliorer les performances du modèle.
    5. **Réduction de dimensionnalité** :
        - Utiliser des techniques comme PCA (Principal Component Analysis) pour réduire le nombre de variables ou la sélection de variable (cf. **Sélection des variables**).
    6. **Gestion des valeurs aberrantes** :
        - Identifier et traiter les valeurs aberrantes qui pourraient affecter les performances du modèle.
    7. **Équilibrage des classes** :
        - Si les classes sont déséquilibrées, appliquer des techniques comme le suréchantillonnage ou le sous-échantillonnage pour équilibrer les classes.
    """)

    st.write("Ces étapes permettent de préparer les données pour la modélisation et d'améliorer les performances des modèles de machine learning.")

    st.write("#### Sélection des variables")
    """ Pour répondre au problème du grand nombre de variable, nous avons essayé diverses technique de réduction de dimension: Select K Best, VarianceThreshold et enfin Optuna. Optuna est un framework d’optimisation des hyperparamètres. Dans notre cas, les paramètres à optimiser sont les variables à considérer ou non. Nous avons donc paramétré Optuna de manière à optimiser la précision de nos prédictions en fonction des variables d'entraînement sélectionnées.

    A chaque itération, Optuna sélectionne certaines variables, entraîne un modèle – dans ce cas un modèle XGBClassifier – puis stocke le résultat de la prédiction. Son objectif est de maximiser la précision tout en minimisant le nombre de variables. Pour cela nous avons ajouté une pénalité qui augmente avec le nombre de variables prises en compte.

    Une fois tous les « trials » terminé, voici les variables restantes :"""
    st.markdown("""
    - 'Shot Distance',
    - 'Season Type',
    - 'Shot Zone Basic_In The Paint (Non-RA)',
    - 'Shot Zone Basic_Right Corner 3',
    - 'Shot Zone Area_Right Side(R)',
    - 'Shot Zone Range_8-16 ft.',
    - 'at_home',
    - 'PREVIOUS_OFF_MISSED',
    - 'Age',
    - 'ASTM',
    - 'ORBM',
    - 'FT%',
    - 'height',
    - 'weight',
    - 'C',
    - 'SG-PG',
    - 'E_DEF_RATING',
    - 'PCT_AREA',
    - 'DETAILLED_SHOT_TYPE_JUMP SHOT'
    """)

    """
    Soit 19 variables contre 63 ! à ce stade la précision du modèle xgboost utilisé avec Optuna est de 67.79%.
    """
#########################################################################################################################################
#                                                           PAGE MODELISATION                                                           #
#########################################################################################################################################
elif page == pages[4]:
    st.write("# Modélisation")
    st.title("Présentation Interactive du Modèle")
    st.write("Explorons les performances de nos modèles Machine Learning et Deep Learning sur notre jeu de données.")
    st.write("Commençons tout d'abord par regarder les données à notre disposition")

    if st.checkbox("Afficher les données brutes"):
        st.write(df_all_shots.head())

    # Sélection de colonnes à explorer
    selected_cols = st.multiselect("Sélectionnez les colonnes à explorer", options=df_all_shots.columns,
                                   default=['Shot Distance', 'Season Type', 'Shot Zone Basic_In The Paint (Non-RA)',
                                            'Shot Zone Basic_Right Corner 3','Shot Zone Area_Right Side(R)',
                                            'Shot Zone Range_8-16 ft.','at_home','PREVIOUS_OFF_MISSED',
                                            'Age','ASTM','ORBM','FT%','height','weight','C','SG-PG','E_DEF_RATING',
                                            'PCT_AREA','DETAILLED_SHOT_TYPE'])
    if selected_cols:
        st.write(df_all_shots[selected_cols].describe())

    model_choice = st.radio("Choisissez le modèle à explorer", ("Machine Learning", "Deep Learning"))

    # Paramètres et informations spécifiques au modèle sélectionné
    if model_choice == "Machine Learning":
        st.subheader("Modèles de Machine Learning")
        """
        Tous les algorithmes testés sont relativement équivalents avec une précision sur l’ensemble de test entre 0.67 et 0.69. \n
        L’arbre de décision et les forêts aléatoires offrent des résultats similaires, nous poursuivrons cependant avec les forêts aléatoires qui sont basées sur les arbres de décisions et permettent de réduire le risque de surapprentissage.
        Le modèle XGBoost Classifier donne les meilleurs résultats à la fois en précision et en temps d'entraînement. \n

        Pour notre cas d'usage, les modèles basés sur les arbres de décisions semblent plus appropriés.
        """
        st.image("src/streamlit/figures/ML_results.png")
        # choix du model de machine learning
        model_choice1 = st.selectbox("Choisissez votre modèle de Machine learning parmi ceux de la liste suivante:",
                                    ["Random Forest", "XGBoost"])

        #présentation des hyperparamètre du model de machine learning
        #random forest
        if model_choice1 == "Random Forest":
            #st.subheader("Random Forest")
            # Paramètres fictifs du modèle de machine learning
            ml_config1 = {
                "Modèle": "Random Forest",
                "n_estimators": 311,
                "max_depth": 14,
                "min_samples_split": 23,
                "min_samples_leaf": 4,
                "Criterion": "gini"
            }
            st.json(ml_config1)  # Affiche la configuration spécifique au modèle ML

            accuracy_ml1 = 0.6831
            st.write(f"Précision du modèle Random Forest : {accuracy_ml1}")

        if model_choice1 == "XGBoost":
            #st.subheader("XGBoost")
            # Paramètres fictifs du modèle de machine learning
            ml_config2 = {
                "Modèle": "XGBoost",
                "booster": "dart",
                "subsample": 0.7152,
                "max_depth": 9,
                "colsample_bytree": 0.8437,
                "min_child_weight": 9,
                "grow_policy": "lossguide"
            }
            st.json(ml_config2)  # Affiche la configuration spécifique au modèle ML


            accuracy_ml2 = 0.6825
            st.write(f"Précision du modèle XGBoost : {accuracy_ml2}")


    #choix d'un model de DL
    if model_choice == "Deep Learning":
        st.subheader("Modèles de Deep Learning")

        # choix du model de deep learning
        model_choice2 = st.selectbox("Choisissez votre modèle de deep learning parmi ceux de la liste suivante:",
                                    ["LSTM", "LeNet"])

        #présentation des hyperparamètre du model de deep learning
        #random forest
        if model_choice2 == "LSTM":
            # Paramètres fictifs du modèle de machine learning
            dl_config1 = {
                "Modèle": "Réseau LSTM",
                "Nombre de couches": 3,
                "Taille des couches": [64, 32, 1],
                "Fonction d'activation": "relu",
                "Taux d'apprentissage": 0.001
            }
            st.json(dl_config1)  # Affiche la configuration spécifique au modèle ML

            accuracy_dl1 = 0.6334
            st.write(f"Précision du modèle LSTM : {accuracy_dl1}")

        if model_choice2 == "LeNet":
            # Paramètres fictifs du modèle de machine learning
            dl_config2 = {
                "Modèle": "LeNet",
                "Nombre de couches": 4,
                "Taille des couches": [32, 64, 128, 1],
                "Fonction d'activation": "relu",
                "Taux d'apprentissage": 0.001
            }
            st.json(dl_config2)  # Affiche la configuration spécifique au modèle ML

            st.write("Data set avec 19 variables")
            st.image("src/streamlit/figures/Capture_Accuracy_Loss_epochs100_LeNet_19Variables.png")
            accuracy_dl2 = 0.6583
            st.write(f"Précision du modèle LeNet : {accuracy_dl2}")
            st.write("validation Accuracy: 0.6583, Training Accuracy: 0.6599 , un gap faible (≈ 0.16%) ")

             st.write("Data set avec 73 variables")
            st.image("src/streamlit/figures/Capture_Accuracy_Loss_epochs100_LeNet_Original_Updated.png")
            accuracy_dl2 = 0.6946
            st.write(f"Précision du modèle LeNet : {accuracy_dl2}")
            st.write("validation Accuracy: 0.6946, Training Accuracy: 0.7027 , gap modéré (≈ 0.81%)  ")


#########################################################################################################################################
#                                                           PAGE DEMO                                                                   #
#########################################################################################################################################

elif page == pages[5]:
    st.write("# Démo")
    @st.cache_data
    def load_demo_data():
        df_demo = pd.read_csv("data/final/data_with_all_columns.csv")
        return df_demo.reset_index(drop=True)

    df_demo = load_demo_data()

    # 1. Choose player
    blank = ['Choisir un joueur']
    liste_joueurs = df_demo['PLAYER1_NAME'].unique()
    blank.extend(liste_joueurs)

    # Add a selectbox to choose a player:
    result_joueur = st.selectbox(
        'Choisir un joueur', blank
    )

    if result_joueur != '':
        # Récupérer les saisons disponibles pour le joueur sélectionné
        annees_disponibles = df_demo['Year'].loc[df_demo['PLAYER1_NAME'] == result_joueur].unique()

        # Sidebar pour sélectionner la saison
        result_annee = st.selectbox("Choisir une saison:", annees_disponibles)

    # Filtre sur les coordonnées X
    if result_annee != '':
        X_locations = df_demo['X Location'].loc[(df_demo['PLAYER1_NAME']==result_joueur) \
                                                & (df_demo['Year']==result_annee)].unique()
        result_x_loc = st.selectbox("Choisir une coordonnée X", sorted(X_locations))

    # Filtre sur les coordonnées Y
    if result_x_loc != '':
        Y_locations = df_demo['Y Location'].loc[(df_demo['PLAYER1_NAME']==result_joueur) \
                                                & (df_demo['Year']==result_annee) \
                                                & (df_demo['X Location']==result_x_loc)].unique()
        result_y_loc = st.selectbox("Choisir une coordonnée Y", sorted(Y_locations))

    # Afficher les données filtrées
    if result_y_loc != '':
        df_to_predict = df_demo.loc[(df_demo['PLAYER1_NAME']==result_joueur) \
                                            & (df_demo['Year']==result_annee) \
                                            & (df_demo['X Location']==result_x_loc) \
                                            & (df_demo['Y Location']==result_y_loc)]

        columns_to_show = ['PLAYER1_NAME', 'Year', 'Shot Distance', 'X Location', 'Y Location']
        st.dataframe(df_to_predict[columns_to_show])
        predict_button = st.button("Prédire !")

    # Prédiction
    if predict_button:
        predict_columns = ['Shot Distance', 'Season Type', 'Shot Zone Basic_In The Paint (Non-RA)',
            'Shot Zone Basic_Right Corner 3', 'Shot Zone Area_Right Side(R)', 'Shot Zone Range_8-16 ft.',
            'at_home', 'PREVIOUS_OFF_MISSED', 'Age', 'ASTM', 'ORBM', 'FT%', 'height', 'weight', 'C', 'SG-PG',
            'E_DEF_RATING', 'PCT_AREA', 'DETAILLED_SHOT_TYPE_JUMP SHOT']

        predictions = []
        for row in df_to_predict.iterrows():
            predictions.append(predict(row[1][predict_columns]))
        df_to_predict['Prediction scores'] = predictions
        df_to_predict['Prediction'] = df_to_predict['Prediction scores'].apply(lambda x: 1 if x >= 0.5 else 0)
        def highlight_row(row):
            if row['Prediction'] == row['target']:
                return ['background-color: #a9dbb7'] * len(row)
            else:
                return ['background-color: #d4a1a3'] * len(row)

        st.dataframe(df_to_predict[columns_to_show + ['Prediction scores', 'Prediction', 'target']].style.apply(highlight_row, axis=1))


#########################################################################################################################################
#                                                           PAGE CONCLUSION                                                             #
#########################################################################################################################################

elif page == pages[6]:
    st.title("Conclusion")
    """
    Notre étude a permis de développer plusieurs modèles prédictifs atteignant une accuracy moyenne de 68%. Ces résultats sont significativement supérieurs à une classification aléatoire ou à une prédiction naïve basée uniquement sur la distribution des tirs ratés (56%), démontrant ainsi la pertinence de notre approche.

    Néanmoins, plusieurs pistes d'amélioration se dégagent pour optimiser ces performances :

    - L'intégration de nouvelles variables comme la distance défenseur-tireur
    - La prise en compte de l'angle de tir
    - L'enrichissement des données avec d'autres métriques contextuelles

    On pourrait aussi améliorer notre étude en ajoutant une partie d'interprétabilité des résultats grace aux SHAP values par exemple.    
    
    Il est important de noter que même dans des conditions optimales, une prédiction parfaite reste illusoire. L'exemple des lancers francs est particulièrement éloquent : même dans cette situation de jeu parfaitement contrôlée, les meilleurs joueurs n'atteignent qu'environ 80% de réussite, illustrant la part incompressible d'incertitude inhérente au basketball.

    D'un point de vue métier, notre étude offre deux contributions majeures :
        
    1. Une analyse approfondie des patterns de tir qui permet aux équipes d'optimiser leur stratégie défensive en fonction des zones de prédilection et d'efficacité de chaque joueur
    2. Un modèle prédictif qui, bien qu'imparfait, fournit un outil supplémentaire d'aide à la décision pour les analystes et les coaches.
    """
