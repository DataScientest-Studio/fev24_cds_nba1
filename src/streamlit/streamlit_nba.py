#importer les packages

import pandas as pd
import streamlit as st
import os


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df_fin = pd.read_csv('data/processed/stat_joueurs_streamlit.csv')

df10 = pd.read_csv('data/processed/all_shots_2000-2020_shot_types_categorized.csv')

st.sidebar.title("Sommaire")

pages = ["Description du projet", "Exploration des données", "Analyse de données", "Stat des  joueurs", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

blank = ['Choisi un joueur']
liste_joueurs = df_fin['Player'].unique()
blank.extend(liste_joueurs)

# Add a selectbox to the sidebar:

result_joueur = st.sidebar.selectbox(
    'Choisi un joueur', blank
)
#result_annee = df_fin['Year'].loc[df_fin['Player'] == result_joueur]
#result_annee = None
# choix de l'année de jeu



if result_joueur != '':
    # Récupérer les années disponibles pour le joueur sélectionné
    annees_disponibles = df_fin['Year'].loc[df_fin['Player'] == result_joueur].unique()


    # Sidebar pour sélectionner l'année
    result_annee = st.sidebar.selectbox("Choisissez une année:", annees_disponibles)

    # Afficher l'année sélectionnée
    st.sidebar.write(f"L'année sélectionnée est: {result_annee}")
    # Affichez le bouton de choix de
    if st.sidebar.button('Choisir la page'):
        st.sidebar.write("Bouton de choix de page cliqué")
        page = pages[3]


if page == pages[0] :
    st.write('### Description du projet')
    st.write("Les sports américains sont très friands de statistiques, et la NBA (National Basketball Association) ne fait pas exception à la règle. ")

    st.write("Le développement constant des nouvelles technologies et des outils numériques, permet désormais de suivre en temps réel les déplacements de tous les joueurs sur un terrain de basket. Les données recueillies sont ainsi très nombreuses et riches.")

    st.write("Le but de ce projet est de : ")

    st.write("    1.  Comparer les tirs (fréquence et efficacité au tir par situation de jeu et par localisation sur le terrain) de 20 des meilleurs joueurs de NBA du 21ème siècle.")

    st.write("    2.  Pour chacun de ces 20 joueurs encore actifs aujourd’hui (de LeBron James à Giannis Antetokounmpo), estimer à l’aide d’un modèle la probabilité qu’a leur tir de rentrer dans le panier, en fonction de différentes métriques.")

    st.write("### Problématique")

    st.write("Nous sommes face à un problème de classification, nous devons déterminer si le tir rentre ou non dans le panier en nous basant sur les données de localisation du tir, les performances du joueur et de son équipe, la situation de jeu. ")

    st.write("### Datasets")

    st.write("L’ensemble de ces données peuvent également être récupérées via des API qui scrappent NBA.com, le site officiel.")

    st.write("Notamment https://github.com/swar/nba_api, grâce aux contributeurs que nous pourrons utiliser afin d’obtenir les données actualisées depuis le site NBA.com.")

    st.write("L’API fournit de nombreuses méthodes permettant de récupérer les joueurs et leur stats, l’historique des matches et le play by play pour chacun d’eux.")

    st.write("Sur ce bonne exploration")

    st.image("C:/Users/mboko/PycharmProjects/nba_intro/jordan.jpg")

elif page == pages[1]:
    st.write("### Exploration des données")

    st.write("## A remplir")

elif page == pages[2]:
    st.write("### Analyse de données")

    st.write("## A remplir")

elif page == pages[3]:

    st.write("### Stats des joueurs")

    st.write("Ici nous pouvons voir les statistiques des joueurs ainsi que leurs positions préférentiel sur le terrain")

    st.write("###", result_joueur)
    pts_moyen = df_fin['PTM'].loc[(df_fin['Player'] == result_joueur) & (df_fin['Year'] == result_annee)].unique()[0]

    # Création des colonnes
    col1, col2 = st.columns(2)

    # Contenu de la colonne de gauche
    with col1:
        st.write("Moyenne de points marqués par match sur la saison")

    df10.drop(df10.loc[df10['free_throw'] == 1].index, inplace=True)
    df10['X Location'] = df10['X Location'] / 10
    df10['Y Location'] = df10['Y Location'] / 10

    # Contenu de la colonne de gauche
    def get_image_path(player_name):
        return f"C:/Users/mboko/PycharmProjects/nba_intro/{result_joueur}.png"


    # Obtenir le chemin de l'image
    image_path = get_image_path(result_joueur)

    with col2:
        # Vérifier si le fichier existe
        if os.path.exists(image_path):
            st.image(image_path)
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
    df10['X Location'] = df10['X Location'].apply(custom_round)
    df10['Y Location'] = df10['Y Location'].apply(custom_round)

    df_n = df10[['PLAYER1_NAME', 'Year', 'X Location', 'Y Location', 'target']]

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
    import matplotlib.patches as patches
    import matplotlib.image as mpimg


    def draw_basketball_court(joueur, annee):
        fig, ax = plt.subplots(figsize=(12, 8))
        # Changer la couleur de fond de la figure
        background_img = mpimg.imread(
            'C:/Users/mboko/PycharmProjects/nba_intro/Hardwood basketball court floor viewed from above _ My Affordable Flooring.jpg')

        # Afficher l'image de fond
        ax.imshow(background_img, extent=[0, 50, 0, 94], aspect='auto')

        # Dimensions du terrain de basket-ball en pieds
        court_length_ft = 94
        court_width_ft = 50

        # Ligne de fond
        ax.plot([0, court_width_ft], [0, 0], linewidth=2, color='black')
        ax.plot([0, court_width_ft], [court_length_ft, court_length_ft], linewidth=2, color='black')

        # Lignes de côté
        ax.plot([0, 0], [0, court_length_ft], linewidth=2, color='black')
        ax.plot([court_width_ft, court_width_ft], [0, court_length_ft], linewidth=2, color='black')

        # arceau
        basket_circle_radius_ft = 1  # Rayon du cercle central
        ax.add_patch(
            patches.Circle((court_width_ft / 2, 5.5), basket_circle_radius_ft, color='black', fill=False, linewidth=1))
        ax.add_patch(
            patches.Circle((court_width_ft / 2, 88.5), basket_circle_radius_ft, color='black', fill=False, linewidth=1))

        # cercle central
        center_circle_radius_ft = 6  # Rayon du cercle central
        ax.add_patch(patches.Circle((court_width_ft / 2, court_length_ft / 2), center_circle_radius_ft, color='black',
                                    fill=False, linewidth=2))

        # cercle ligne des lancers francs
        center_circle_radius_ft = 6  # Rayon du cercle central
        ax.add_patch(
            patches.Circle((court_width_ft / 2, 19), center_circle_radius_ft, color='black', fill=False, linewidth=2))
        ax.add_patch(
            patches.Circle((court_width_ft / 2, 75), center_circle_radius_ft, color='black', fill=False, linewidth=2))

        # position du panier
        basket_dist = 4  # Distance entre le panier et la ligne de fond

        # Ligne des 3 points
        three_point_dist_ft = 23.9  # Distance du panier
        corner_three_dist_ft = 22  # Distance du coin
        three_point_radius_ft = 22 / 12  # Rayon du cercle
        ax.plot([0, 0], [0, three_point_dist_ft], linewidth=2, color='black')
        ax.plot([court_width_ft, court_width_ft], [4, three_point_dist_ft], linewidth=2, color='black')
        ax.plot([0, 0], [court_length_ft - three_point_dist_ft, court_length_ft], linewidth=2, color='black')
        # ax.plot([court_width_ft, court_width_ft], [court_length_ft-three_point_dist_ft, court_length_ft], linewidth=2, color='black')
        ax.add_patch(
            patches.Arc((court_width_ft / 2, three_point_dist_ft / (3.14 * 2)), width=48.3, height=50, theta1=25,
                        theta2=155, linewidth=2, color='black'))
        ax.add_patch(
            patches.Arc((court_width_ft / 2, court_length_ft - three_point_dist_ft / (3.14 * 2)), width=48.3, height=50,
                        theta1=205, theta2=335, linewidth=2, color='black'))

        # Raquettes
        paint_width_ft = 16  # Largeur de la raquette
        paint_length_ft = 19  # Longueur de la raquette
        ax.plot([17, 17 + paint_width_ft], [0, 0], linewidth=2, color='black')
        ax.plot([17, 17 + paint_width_ft], [paint_length_ft, paint_length_ft], linewidth=2, color='black')
        ax.plot([17, 17], [0, paint_length_ft], linewidth=2, color='black')
        ax.plot([17 + paint_width_ft, 17 + paint_width_ft], [0, paint_length_ft], linewidth=2, color='black')

        ax.plot([17, 17 + paint_width_ft], [75, 75], linewidth=2, color='black')
        ax.plot([17, 17 + paint_width_ft], [paint_length_ft, paint_length_ft], linewidth=2, color='black')
        ax.plot([17, 17], [75, court_length_ft], linewidth=2, color='black')
        ax.plot([17 + paint_width_ft, 17 + paint_width_ft], [75, court_length_ft], linewidth=2, color='black')

        # ligne médiane
        ax.plot([court_width_ft, 0], [47, 47], linewidth=2, color='black')

        # ligne planche
        ax.plot([22, 28], [4, 4], linewidth=2, color='black')
        ax.plot([22, 28], [90, 90], linewidth=2, color='black')

        # ligne 3pts side line bas
        ax.plot([3, 3], [0, 14], linewidth=2, color='black')
        ax.plot([47, 47], [0, 14], linewidth=2, color='black')

        # ligne 3pts side line haut
        ax.plot([3, 3], [court_length_ft, 80], linewidth=2, color='black')
        ax.plot([47, 47], [court_length_ft, 80], linewidth=2, color='black')

        # ax.add_patch(patches.Arc((court_width_ft/2, 9/(3.14*2)), width=10, height=10, theta1=0, theta2=180, linewidth=2, color='black'))

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


        # Effacer les axes
        # ax.axis('off')

        #plt.show()
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

elif page == pages[4]:
    st.write("### Modélisation")

    st.write("## A remplir")

