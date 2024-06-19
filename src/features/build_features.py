import pandas as pd
import numpy as np


# Liste des 20 meilleurs joueurs qui seront étudiés
BEST_PLAYERS =['Kobe Bryant', 'Lebron James', 'Stephen Curry', 'Kevin Durant', 'Dwayne Wade', 'Dirk Nowitski', 'Tim Duncan', "Shaquille O'Neal", "Steve Nash", "Kawhi Leonard", "James Harden", "Jason Kidd", "Allen Iverson", "Chris Webber", "Kevin Garnett", "Paul Pierce", "Giannis Antetokounmpo", "Jimmy Butler", "Russell Westbrook", "Dwight Howard"]


def drop_players(data):
    """ Detele players that are not part of the study
    Args: 
        data : DataFrame
    Returns:
        data : DataFrame
    """
    data = data[data.PLAYER1_NAME.isin(BEST_PLAYERS)]

    return data

def drop_actions(data):
    """ Delete actions that are not shots (rebounds, violations, fouls, timeouts, etc.)
    Args: 
        data : DataFrame
    Returns:
        data : DataFrame
    """
    data = data[data.EVENTMSGTYPE < 4]
    return data 


def drop_assists(data):
    """ Delete actions where PLAYER2 is not null, it reprensents the assiting player so it means the shot is made
    Args: 
        data : DataFrame
    Returns:
        data : DataFrame
    """
    data = data[data.PLAYER2_ID == 0]
    return data 


def drop_blocks(data):
    """ Delete actions where PLAYER3 is not null, it reprensents the blocking player so it means the shot is not made
    Args: 
        data : DataFrame
    Returns:
        data : DataFrame
    """
    data = data[data.PLAYER3_ID == 0]
    return data


def drop_columns(data, columns):
    """ Delete unnecessary columns
     Args: 
        data : DataFrame
        columns : list of columns to drop
    Returns:
        data : DataFrame
    """

    return data.drop(columns, axis = 1)

