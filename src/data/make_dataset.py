import pandas as pd


BEST_PLAYERS =['Kobe Bryant', 'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Dwyane Wade', 'Dirk Nowitzki', 'Tim Duncan', "Shaquille O'Neal", "Steve Nash", "Kawhi Leonard", "James Harden", "Jason Kidd", "Allen Iverson", "Chris Webber", "Kevin Garnett", "Paul Pierce", "Giannis Antetokounmpo", "Jimmy Butler", "Russell Westbrook", "Dwight Howard"]
PLAYERS_DICT = {
    'K. Bryant':'Kobe Bryant', 
    'L. James':'LeBron James', 
    'S. Curry':'Stephen Curry', 
    'K. Durant':'Kevin Durant', 
    'D. Wade':'Dwyane Wade', 
    'D. Nowitzki':'Dirk Nowitzki', 
    'T. Duncan':'Tim Duncan', 
    "S. O'Neal":"Shaquille O'Neal", 
    'S. Nash':"Steve Nash", 
    'K. Leonard':"Kawhi Leonard", 
    'J. Harden':"James Harden", 
    'J. Kidd':"Jason Kidd", 
    'A. Iverson':"Allen Iverson", 
    'C. Webber':"Chris Webber", 
    'K. Garnett':"Kevin Garnett", 
    'P. Pierce':"Paul Pierce", 
    'G. Antetokounmpo':"Giannis Antetokounmpo", 
    'J. Butler':"Jimmy Butler", 
    'R. Westbrook':"Russell Westbrook", 
    'D. Howard':"Dwight Howard"
}
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

def update_freethrow_outcome(data):
    """ Update the outcome of a free throw, checks if SCORE is updated. The score is updated only if the goal is made
    Args: 
        data : DataFrame
    Returns:
        data : DataFrame
    """      
    data.loc[(data.EVENTMSGTYPE==3) & (data.SCORE.isna()), 'EVENTMSGTYPE'] = 2  # MISSED
    data.loc[(data.EVENTMSGTYPE==3) & (data.SCORE.notna()), 'EVENTMSGTYPE'] = 1 # MADE
    return data


def create_3pt_feature(data):
    data.loc[(data.HOMEDESCRIPTION.str.contains('3PT')) | (data.VISITORDESCRIPTION.str.contains('3PT')), '3PT'] = 1
    data.loc[data['3PT'].isna(), '3PT'] = 0
    return data

def create_jumpshot_feature(data):    
    data.loc[(data.HOMEDESCRIPTION.str.contains('Jump Shot')) | (data.VISITORDESCRIPTION.str.contains('Jump Shot')), 'jump_shot'] = 1
    data.loc[data['jump_shot'].isna(), 'jump_shot'] = 0
    return data

def create_layup_feature(data):    
    data.loc[(data.HOMEDESCRIPTION.str.contains('Layup')) | (data.VISITORDESCRIPTION.str.contains('Layup')), 'layup_shot'] = 1
    data.loc[data['layup_shot'].isna(), 'layup_shot'] = 0
    return data

def create_dunk_feature(data):    
    data.loc[(data.HOMEDESCRIPTION.str.contains('Dunk')) | (data.VISITORDESCRIPTION.str.contains('Dunk')), 'dunk_shot'] = 1
    data.loc[data['dunk_shot'].isna(), 'dunk_shot'] = 0
    return data

def create_hook_feature(data):    
    data.loc[(data.HOMEDESCRIPTION.str.contains('Hook')) | (data.VISITORDESCRIPTION.str.contains('Hook')), 'hook_shot'] = 1
    data.loc[data['hook_shot'].isna(), 'hook_shot'] = 0
    return data

def create_freethrow_feature(data):    
    data.loc[(data.HOMEDESCRIPTION.str.contains('Free Throw')) | (data.VISITORDESCRIPTION.str.contains('Free Throw')), 'free_throw'] = 1
    data.loc[data['free_throw'].isna(), 'free_throw'] = 0
    return data

def create_previous_actions_features(data):
    # previous action is an offensive rebound
    data.loc[(data['EVENTMSGTYPE'] < 4) & 
       (data['PLAYER1_TEAM_ID'] == data['PLAYER1_TEAM_ID'].shift(1))  & 
       (data['PLAYER1_TEAM_ID'].shift(1) == data['PLAYER1_TEAM_ID'].shift(2)) & 
       (~data['EVENTMSGTYPE'].shift(1).isin([6,8,10])), 'PREVIOUS_OFF_REBOUND'] = True 
       
    data['PREVIOUS_OFF_REBOUND'] = data['PREVIOUS_OFF_REBOUND'].fillna(False)

    # previous action is a defensive rebound
    data.loc[(data['EVENTMSGTYPE'] < 4) & 
        (data['PLAYER1_TEAM_ID'] == data['PLAYER1_TEAM_ID'].shift(1))  & 
        (data['PLAYER1_TEAM_ID'].shift(1) != data['PLAYER1_TEAM_ID'].shift(2)) & 
        (data['EVENTMSGTYPE'].shift(1) == 4), 'PREVIOUS_DEF_REBOUND'] = True

    data['PREVIOUS_DEF_REBOUND'] = data['PREVIOUS_DEF_REBOUND'].fillna(False)

    # previous action is a turnover
    data.loc[(data['EVENTMSGTYPE'] < 4) & 
        (data['PLAYER1_TEAM_ID'] == data['PLAYER1_TEAM_ID'].shift(1))  & 
        (data['EVENTMSGTYPE'].shift(1) == 5), 'PREVIOUS_OFF_TURNOVER'] = True 
        
    data['PREVIOUS_OFF_TURNOVER'] = data['PREVIOUS_OFF_TURNOVER'].fillna(False)

    # previous action is a field goal missed or free throw
    data.loc[(data['EVENTMSGTYPE'] < 4) & 
        (data['PLAYER1_TEAM_ID'] == data['PLAYER1_TEAM_ID'].shift(1))  & 
        (data['EVENTMSGTYPE'].shift(1).isin([2, 3])), 'PREVIOUS_OFF_MISSED'] = True 
        
    data['PREVIOUS_OFF_MISSED'] = data['PREVIOUS_OFF_MISSED'].fillna(False)

    # Explain Previous actions details
    eventmsgtypes = {
        1: "FIELD_GOAL_MADE",
        2 : "FIELD_GOAL_MISSED",
        3 : "FREE_THROW",
        4 : "REBOUND",
        5 : "TURNOVER",
        6 : "FOUL",
        7 : "VIOLATION",
        8 : "SUBSTITUTION",
        9 : "TIMEOUT",
        10 : "JUMP_BALL",
        11 : "EJECTION" ,
        12 : "PERIOD_BEGIN" ,
        13 : "PERIOD_END" 
    }

    data['PREVIOUS_EVENTMSGTYPE'] = data.EVENTMSGTYPE.shift(1)
    data.PREVIOUS_EVENTMSGTYPE = data.PREVIOUS_EVENTMSGTYPE.replace(eventmsgtypes)

    return data

def update_shot_type(data):    
    data.loc[(data.DETAILLED_SHOT_TYPE != 'JUMP SHOT') & (data.DETAILLED_SHOT_TYPE != "FREE THROW"), 'DETAILLED_SHOT_TYPE'] = 'OTHER'
    return data

def detail_shot_type(data):
    eventnames= {
        102:'3PT DRIVING FLOATING BANK JUMP SHOT',
        101:'3PT DRIVING FLOATING JUMP SHOT',
        63:'3PT FADEAWAY JUMPER',
        78:'3PT FLOATING JUMP SHOT',
        66:'3PT JUMP BANK SHOT',
        1:'3PT JUMP SHOT',
        10:'FREE THROW',
        11:'FREE THROW',
        12:'FREE THROW',
        13:'FREE THROW',
        14:'FREE THROW',
        15:'FREE THROW',
        18:'FREE THROW',
        19:'FREE THROW',
        56:'RUNNING HOOK SHOT',
        59:'FINGER ROLL',
        40:'LAYUP',
        45:'JUMP SHOT',
        61:'DRIVING FINGER ROLL',
        79:'3PT PULLUP JUMP SHOT',
        2:'3PT RUNNING JUMP SHOT',
        103:'3PT RUNNING PULL',
        104:'3PT STEP BACK BANK JUMP SHOT',
        80:'3PT STEP BACK JUMP SHOT',
        86:'3PT TURNAROUND FADEAWAY',
        105:'3PT TURNAROUND FADEAWAY BANK JUMP SHOT',
        47:'3PT TURNAROUND JUMP SHOT',
        52:'ALLEY OOP DUNK',
        43:'ALLEY OOP LAYUP',
        108:'CUTTING DUNK SHOT',
        99:'CUTTING FINGER ROLL LAYUP SHOT',
        98:'CUTTING LAYUP SHOT',
        93:'DRIVING BANK HOOK SHOT',
        9:'DRIVING DUNK',
        75:'DRIVING FINGER ROLL LAYUP',
        57:'DRIVING HOOK SHOT',
        6:'DRIVING LAYUP',
        109:'DRIVING REVERSE DUNK SHOT',
        73:'DRIVING REVERSE LAYUP',
        7:'DUNK',
        71:'FINGER ROLL LAYUP',
        67:'HOOK BANK SHOT',
        3:'HOOK SHOT',
        5:'LAYUP',
        87:'PUTBACK DUNK',
        72:'PUTBACK LAYUP',
        51:'REVERSE DUNK',
        44:'REVERSE LAYUP',
        106:'RUNNING ALLEY OOP DUNK SHOT',
        100:'RUNNING ALLEY OOP LAYUP SHOT',
        50:'RUNNING DUNK',
        76:'RUNNING FINGER ROLL LAYUP',
        41:'RUNNING LAYUP',
        110:'RUNNING REVERSE DUNK SHOT',
        74:'RUNNING REVERSE LAYUP',
        107:'TIP DUNK SHOT',
        97:'TIP LAYUP SHOT',
        96:'TURNAROUND BANK HOOK SHOT',
        58:'TURNAROUND HOOK SHOT',
        42:'LAYUP',
        49:'DRIVING_DUNK',
        46:'RUNNING_JUMP_SHOT',
        8:'SLAM_DUNK',
        4:'TIP_SHOT',
        16: "FREE THROW",
        17: "FREE THROW",
        55: "HOOK SHOT",                         
        48: "DUNK SHOT",                        
        60: "RUNNING FINGER ROLL",                         
        53: "TIP SHOT",
    }

    data["DETAILLED_SHOT_TYPE"] = data.EVENTMSGACTIONTYPE.replace(eventnames)

    return data

def clean_data(data):

    # add feature "opponent team"
    games = data[['GAME_ID', 'PLAYER1_TEAM_ABBREVIATION']].dropna().drop_duplicates()
    for _, game in games.iterrows():
        data.loc[(data.GAME_ID == game.GAME_ID) & (data.PLAYER1_TEAM_ABBREVIATION != game.PLAYER1_TEAM_ABBREVIATION), 'OPPONENT_TEAM'] = game.PLAYER1_TEAM_ABBREVIATION

    # Change index
    data = data.set_index(['GAME_ID', 'EVENTNUM'])

    # Create feature at_home to see if players perform better when they're at home
    data.loc[data.HOMEDESCRIPTION.notna(), 'at_home'] = 1
    data.loc[data.HOMEDESCRIPTION.isna(), 'at_home'] = 0

    data.loc[data.HOMEDESCRIPTION.isna(), 'HOMEDESCRIPTION'] = ''
    data.loc[data.VISITORDESCRIPTION.isna(), 'VISITORDESCRIPTION'] = ''

    data = create_previous_actions_features(data)

    data = drop_players(data)
    data = drop_actions(data)
    data = update_freethrow_outcome(data)
    
    # Transform EVENTMSGTYPE to have 0 = MISSED and 1 = MADE
    data.loc[data.EVENTMSGTYPE==2, 'EVENTMSGTYPE'] = 0
    
    # Create features from HOMEDESCRIPTION and VISITORDESCRIPTION
    data = create_3pt_feature(data)
    data = create_jumpshot_feature(data)
    data = create_layup_feature(data)
    data = create_dunk_feature(data)
    data = create_hook_feature(data)
    data = create_freethrow_feature(data)
    data = detail_shot_type(data)

    # Transform PCTIMESTRING as time
    data.PCTIMESTRING = pd.to_datetime(data.PCTIMESTRING, format="%M:%S") 

    # Create features from PCTIMESTRING
    data['minutes_left'] = data['PCTIMESTRING'].dt.minute
    data['seconds_left'] = data['PCTIMESTRING'].dt.minute*60 + data['PCTIMESTRING'].dt.second
    
    # Drop columns    
    data = data.drop(['HOMEDESCRIPTION', 'NEUTRALDESCRIPTION', 'PERSON2TYPE', 'PERSON3TYPE', 'PLAYER2_ID', 'PLAYER2_NAME', 'PLAYER2_TEAM_ABBREVIATION', \
                'PLAYER2_TEAM_CITY', 'PLAYER2_TEAM_ID', 'PLAYER1_TEAM_NICKNAME', 'PLAYER1_ID', 'PLAYER1_TEAM_CITY', 'PLAYER1_TEAM_ID','PLAYER2_TEAM_NICKNAME', \
                'PLAYER3_ID', 'PLAYER3_NAME', 'PLAYER3_TEAM_ABBREVIATION', 'PLAYER3_TEAM_CITY', 'PLAYER3_TEAM_ID', 'PLAYER3_TEAM_NICKNAME', 'SCORE', \
                'VISITORDESCRIPTION', 'WCTIMESTRING', 'EVENTMSGACTIONTYPE', 'PCTIMESTRING', 'PERSON1TYPE'], axis = 1)

    # Rename target column
    data = data.rename({'EVENTMSGTYPE': 'target'}, axis=1)

    data = update_shot_type(data)

    return data

def main():
    # Load data
    pct_area = pd.read_csv("data/processed/pourcentage_par_zone.csv", index_col=0)
    pct_action = pd.read_csv("data/processed/pourcentage_par_action_precedente.csv", index_col=0)

    shot_locations = pd.read_csv("data/processed/Shot_Locations_top_20_players_2000to2020.csv")
    df_players = pd.read_csv("data/processed/stat_joueurs_streamlit.csv", index_col=0)
    metrics = pd.read_csv("data/raw/team_metrics.csv", index_col=0)
    pct_action.dropna(inplace=True)
    pct_area.dropna(inplace=True)

    # Clean all datasets from 2000 to 2019
    files = ['2000-01_pbp.csv','2001-02_pbp.csv','2002-03_pbp.csv','2003-04_pbp.csv','2004-05_pbp.csv',
            '2005-06_pbp.csv','2006-07_pbp.csv','2007-08_pbp.csv','2008-09_pbp.csv','2009-10_pbp.csv',
            '2010-11_pbp.csv','2011-12_pbp.csv','2012-13_pbp.csv','2013-14_pbp.csv','2014-15_pbp.csv',
            '2015-16_pbp.csv','2016-17_pbp.csv','2017-18_pbp.csv','2018-19_pbp.csv', 'missing_pbp_2019-2020.csv',
            'missing_pbp.csv']

    all_data = []

    for file in files:
        data = pd.read_csv("data/raw/" + file, index_col=0)    
        all_data.append(clean_data(data))


    # Concat all play by play data
    all_plays = pd.concat(all_data)
    
    if 'VIDEO_AVAILABLE_FLAG' in all_plays.columns:
        all_plays.drop('VIDEO_AVAILABLE_FLAG', axis=1, inplace=True)
    
    all_plays.reset_index(inplace=True)
    

    # Merge shot location    
    df_merged = shot_locations.merge(all_plays, how='outer', left_on=['Game ID','Game Event ID'], right_on=['GAME_ID','EVENTNUM'])

    # Fill Year feature with data from complete lines 
    games = df_merged[['GAME_ID', 'Year']].drop_duplicates().dropna()

    for _, game in games.iterrows():
        df_merged.loc[df_merged.GAME_ID==game['GAME_ID'], 'Year']=game['Year']
    
    # Keep only the first 4 caracters to have the year
    df_players['Year'] = df_players['Year'].astype(str).str.slice(0, 4).astype(int)

    # Merge player stats
    data = df_merged.merge(df_players, how='left', left_on=['PLAYER1_NAME', 'Year'], right_on=['Player', 'Year'])

    # drop duplicated columns
    data = data.drop(['Game ID', 'Game Event ID', 'Player Name', 'Player', 'Team Name', 'Player ID', 'Team ID' ], axis = 1)
    
    # update shot location for all free throws 
    data.loc[data['free_throw']==1, 'Shot Zone Basic_Mid-Range'] = True
    data.loc[data['free_throw']==1, 'Shot Zone Area_Center(C)'] = True
    data.loc[data['free_throw']==1, 'Shot Zone Range_8-16 ft.'] = True
    data.loc[data['free_throw']==1, 'Shot Distance'] = 15.0
    data.loc[data['free_throw']==1, 'X Location'] = 0
    data.loc[data['free_throw']==1, 'Y Location'] = 150

    data.fillna({'Shot Zone Basic_Above the Break 3':False}, inplace=True)
    data.fillna({'Shot Zone Basic_Backcourt': False}, inplace=True)
    data.fillna({'Shot Zone Basic_In The Paint (Non-RA)': False}, inplace=True)
    data.fillna({'Shot Zone Basic_Left Corner 3' : False}, inplace=True)
    data.fillna({'Shot Zone Basic_Mid-Range' : False}, inplace=True)
    data.fillna({'Shot Zone Basic_Restricted Area' : False}, inplace=True)
    data.fillna({'Shot Zone Basic_Right Corner 3' : False}, inplace=True)
    data.fillna({'Shot Zone Area_Back Court(BC)' : False}, inplace=True)
    data.fillna({'Shot Zone Area_Center(C)' : False}, inplace=True)
    data.fillna({'Shot Zone Area_Left Side Center(LC)' : False}, inplace=True)
    data.fillna({'Shot Zone Area_Left Side(L)' : False}, inplace=True)
    data.fillna({'Shot Zone Area_Right Side Center(RC)' : False}, inplace=True)
    data.fillna({'Shot Zone Area_Right Side(R)' : False}, inplace=True)
    data.fillna({'Shot Zone Range_16-24 ft.' : False}, inplace=True)
    data.fillna({'Shot Zone Range_24+ ft.' : False}, inplace=True)
    data.fillna({'Shot Zone Range_8-16 ft.' : False}, inplace=True)
    data.fillna({'Shot Zone Range_Back Court Shot' : False}, inplace=True)
    data.fillna({'Shot Zone Range_Less Than 8 ft.' : False}, inplace=True)
    data.fillna({"3P%":0}, inplace=True)

    # drop Shot Made Flag : target is the same with no NAs
    data.drop('Shot Made Flag', axis=1, inplace = True)

    # drop Home Team : PLAYER1_TEAM_ABBREVIATION is the same with no NAs
    data.drop('Home Team', axis=1, inplace = True)
   
    # Merge team stats
    # Fill Season Type and Away Team with data from complete lines
    games = data[['GAME_ID', 'Away Team', 'Season Type']].drop_duplicates().dropna()

    for _, game in games.iterrows():
        data.loc[data.GAME_ID==game['GAME_ID'], 'Away Team']=game['Away Team']
        data.loc[data.GAME_ID==game['GAME_ID'], 'Season Type']=game['Season Type']

    # Ajout du defensive rate de l'équipe opposée
    data = data.merge(metrics[['ABBREVIATION', 'Year', 'E_DEF_RATING']], left_on=['OPPONENT_TEAM', 'Year'], right_on=['ABBREVIATION', 'Year'])
    data.drop('ABBREVIATION', axis=1, inplace=True)

    # Ajout de l'offensive rate de l'équipe qui tire
    data = data.merge(metrics[['ABBREVIATION', 'Year', 'E_OFF_RATING']], left_on=['PLAYER1_TEAM_ABBREVIATION', 'Year'], right_on=['ABBREVIATION', 'Year'])
    data.drop('ABBREVIATION', axis=1, inplace=True)
    data.dropna(inplace=True)

    # Ajout des années d'expérience
    data['YEARS_EXP'] = data['Year']-data['year_start']

    

    # get dummies to match the columns' names of shots
    pct_action = pd.concat([pct_action, pd.get_dummies(pct_action.PREVIOUS, prefix="PREVIOUS")], axis=1)
    pct_area = pd.concat([pct_area, pd.get_dummies(pct_area["Shot Zone"], prefix="Shot Zone", prefix_sep=" ")], axis=1)

    # drop unneeded columns
    pct_action.drop(['PREVIOUS','Total_Target',	'Count'], axis=1, inplace=True)
    pct_area.drop(["Shot Zone",	"Total_Target",	"Count"], axis=1, inplace=True)

    # rename PCT column
    pct_action.rename({'Pourcentage':'PCT_PREV_ACTION'}, axis=1, inplace=True)
    pct_area.rename({'Pourcentage':'PCT_AREA'}, axis=1, inplace=True)

    # merge new columns with the shots dataframe
    data = pd.merge(left=data, right=pct_action, how='left', on=["PLAYER1_NAME",	"Year",	"PREVIOUS_DEF_REBOUND",	"PREVIOUS_OFF_MISSED",	"PREVIOUS_OFF_REBOUND"])
    data = pd.merge(left=data, right=pct_area, how='left', on=["PLAYER1_NAME", "Year", "Shot Zone Basic_Above the Break 3", 
                                                        "Shot Zone Basic_Backcourt", "Shot Zone Basic_In The Paint (Non-RA)", 
                                                        "Shot Zone Basic_Left Corner 3", "Shot Zone Basic_Mid-Range", 
                                                        "Shot Zone Basic_Restricted Area", "Shot Zone Basic_Right Corner 3"])


    data = pd.concat([pd.get_dummies(data.DETAILLED_SHOT_TYPE, prefix="DETAILLED_SHOT_TYPE"), data], axis=1)
    
    # Save full merged file
    data.to_csv("data/final/data_with_all_columns.csv", index=False)

    # Save only optuna columns for training
    optuna_columns = ['Shot Distance',
                    'Season Type',
                    'Shot Zone Basic_In The Paint (Non-RA)',
                    'Shot Zone Basic_Right Corner 3',
                    'Shot Zone Area_Right Side(R)',
                    'Shot Zone Range_8-16 ft.',
                    'at_home',
                    'PREVIOUS_OFF_MISSED',
                    'Age',
                    'ASTM',
                    'ORBM',
                    'FT%',
                    'height',
                    'weight',
                    'C',
                    'SG-PG',
                    'E_DEF_RATING',
                    'PCT_AREA',
                    'DETAILLED_SHOT_TYPE_JUMP SHOT']

    data = data[optuna_columns + ['target']]

    # Save file
    data.to_csv("data/final/data.csv", index=False)

if __name__ == "__main__":
    main()