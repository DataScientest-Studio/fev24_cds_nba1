import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


def draw_basketball_court(joueur, annee):
    fig, ax = plt.subplots(figsize=(8, 8))
    # Changer la couleur de fond de la figure
    background_img = mpimg.imread(
        'reports/pictures/Hardwood basketball court floor viewed from above _ My Affordable Flooring.jpg')

    # background_img = mpimg.imread('reports/figures/saved_court.png')

    # Afficher l'image de fond
    ax.imshow(background_img, extent=[0, 50, 0, 50], aspect='auto')

    # Dimensions du terrain de basket-ball en pieds
    court_length_ft = 47
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


    # cercle central
    center_circle_radius_ft = 6  # Rayon du cercle central
    ax.add_patch(patches.Circle((court_width_ft / 2, court_length_ft), center_circle_radius_ft, color='black',
                                fill=False, linewidth=2))

    # cercle ligne des lancers francs
    center_circle_radius_ft = 6  # Rayon du cercle central
    ax.add_patch(
        patches.Circle((court_width_ft / 2, 19), center_circle_radius_ft, color='black', fill=False, linewidth=2))

    # Ligne des 3 points
    three_point_dist_ft = 23.9  # Distance du panier

    #ax.plot([0, 0], [0, three_point_dist_ft], linewidth=2, color='black')
    #ax.plot([court_width_ft, court_width_ft], [4, three_point_dist_ft], linewidth=2, color='black')
    #ax.plot([0, 0], [court_length_ft - three_point_dist_ft, court_length_ft], linewidth=2, color='black')
    # ax.plot([court_width_ft, court_width_ft], [court_length_ft-three_point_dist_ft, court_length_ft], linewidth=2, color='black')
    ax.add_patch(
        patches.Arc((court_width_ft / 2, three_point_dist_ft / (3.14 * 2)), width=48.3, height=50, theta1=25,
                    theta2=155, linewidth=2, color='black'))
    #ax.add_patch(
        #   patches.Arc((court_width_ft / 2, court_length_ft - three_point_dist_ft / (3.14 * 2)), width=48.3, height=50,
        #              theta1=205, theta2=335, linewidth=2, color='black'))

    # Raquettes
    paint_width_ft = 16  # Largeur de la raquette
    paint_length_ft = 19  # Longueur de la raquette
    ax.plot([17, 17 + paint_width_ft], [0, 0], linewidth=2, color='black')
    ax.plot([17, 17 + paint_width_ft], [paint_length_ft, paint_length_ft], linewidth=2, color='black')
    ax.plot([17, 17], [0, paint_length_ft], linewidth=2, color='black')
    ax.plot([17 + paint_width_ft, 17 + paint_width_ft], [0, paint_length_ft], linewidth=2, color='black')


    # ligne planche
    ax.plot([22, 28], [4, 4], linewidth=2, color='black')
    #ax.plot([22, 28], [90, 90], linewidth=2, color='black')

    # ligne 3pts side line bas
    ax.plot([3, 3], [0, 14], linewidth=2, color='black')
    ax.plot([47, 47], [0, 14], linewidth=2, color='black')

    ax.set_xlim(-0.1,50.2)
    ax.set_ylim(-0.1,50)

    ax.set_axis_off()
    #plt.show()

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('reports/figures/saved_court.png', bbox_inches=extent)

draw_basketball_court(None, None)