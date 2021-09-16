from tqdm import *
import pickle
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_stats(name, stat, series):
    """
    Extract stats through api with time delay in try/except block.
    Retry exceptions, then save locally to pkl file.
    """
    error_log = []
    
    for val in tqdm(series):
        if len(str(val)) > 8:
            try:
                time.sleep(0.5)
                instance = stat(team_id=val)
                data = instance.get_data_frames()[0]
                name = pd.concat([name, data], axis=0, ignore_index=True)
            except:
                error_log.append(val)
        else:
            try:
                time.sleep(0.5)
                instance = stat(player_id=val)
                data = instance.get_data_frames()[0]
                name = pd.concat([name, data], axis=0, ignore_index=True)
            except:
                error_log.append(val)
            
    for val in tqdm(error_log):
        if len(str(val)) > 8:
            time.sleep(0.5)
            instance = stat(team_id=val)
            data = instance.get_data_frames()[0]
            name = pd.concat([name, data], axis=0, ignore_index=True)
        else:
            time.sleep(0.5)
            instance = stat(player_id=val)
            data = instance.get_data_frames()[0]
            name = pd.concat([name, data], axis=0, ignore_index=True)
                
    pickle.dump(name, open('{}.pkl'.format(str(stat).split('.')[4][:-2]),'wb'))

def award_freq(df, awards_df):
    """
    Creates columns for player of the month and player of the week
    """
    week = awards_df[awards_df['DESCRIPTION']=='NBA Player of the Week'].groupby(['PERSON_ID', 'SEASON'])
    month = awards_df[awards_df['DESCRIPTION']=='NBA Player of the Month'].groupby(['PERSON_ID', 'SEASON'])
    
    player_of_week = []
    player_of_month = []
    
    for i in tqdm(range(len(df))):
        player_id = df.loc[i,'PLAYER_ID']
        season_id = df.loc[i, 'SEASON_ID']
        try:
            player_of_week.append(week.get_group((player_id,season_id))['PERSON_ID'].count())
        except:
            player_of_week.append(0)
            
        try:
            player_of_month.append(month.get_group((player_id,season_id))['PERSON_ID'].count())
        except:
            player_of_month.append(0)
    
    df['player_of_week'] = player_of_week
    df['player_of_month'] = player_of_month

def convert_float(df):
    """
    Converts df object types to float
    """
    for i in range(len(df.columns)):
        if df[df.columns[i]].dtype=='object':
            df[df.columns[i]] = df[df.columns[i]].astype('float64')

def drop_corr(df):
    """
    Checks for multicollinearity and drops features with Pearson correlation
    coefficient >0.8.
    """
    df_corr = df.corr()
    corr_names = set()
    for i in range(len(df_corr.columns)):
        for j in range(i):
            if abs(df_corr.iloc[i,j])>=0.8:
                corr_names.add(df_corr.columns[i])
    df.drop(columns=corr_names, axis=1, inplace=True)

def plot_importances(X, y, rf_ins):
    """
    Fits data to randomforestclassifier and
    plots the feature importances
    """
    rf_ins.fit(X, y)
    importance = pd.Series(rf_ins.feature_importances_, index=X.columns).sort_values(ascending=False)
    sns.barplot(x=importance, y=importance.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title('Visualizing Important Features')
    plt.show()

def calc_results(df, pred):
    """
    Converts prediction probabilities to binary true/false and returns labeled
    results dataframe.
    """

    for i in range(len(df)):
        if df.iloc[i, 0] in list(df.groupby('SEASON_ID')['PRED_PROB'].max()):
            pred.append(1)
        else:
            pred.append(0)

    df['PREDICTIONS'] = pred
    
    player = pd.read_pickle('player.pkl')
    player.rename(columns={'PERSON_ID':'PLAYER_ID'}, inplace=True)
    
    df = df.merge(player, how='left', on='PLAYER_ID')
    df = df[(df['PREDICTIONS']==1)|(df['MVP']==1)][['SEASON_ID', 'DISPLAY_FIRST_LAST', 'PREDICTIONS', 'MVP']].sort_values(by='SEASON_ID').reset_index(drop=True)
    
    return df
    return pred
