"""
%pip install -U numpy
%pip install -U pandas
%pip install -U requests
%pip install -U bs4
%pip install -U selenium
%pip install -U matplotlib
%pip install -U seaborn
%pip install -U plotly
%pip install -U scikit-learn
%pip install -U python-dateutil
%pip install -U lxml
%pip install -U colorama
%pip install -U datetime
%pip install selenium
"""

# %%
import sys
import os
import datetime
import shutil                                         as shu
from   colorama               import Style            as st
from   colorama               import Fore             
from   colorama               import Back             as bk

start_time = datetime.datetime.now()
w, h       = shu.get_terminal_size()

def printSeparator():
    print(Fore.GREEN + '-' * w + Fore.WHITE)
    
def logStep(msg):
    l1 = len(msg)
    l2 = w - l1 - 27

    print(Fore.WHITE  + str(datetime.datetime.now()) +  " " + 
          Fore.YELLOW + msg + Fore.WHITE + "-" * l2  )
    sys.stdout.flush()
    
def printDFinfo(name,dfName):
    printSeparator()
    print('Name: ',name)
    printSeparator()
    print(dfName.info())
    printSeparator()
    print(f'Row Count :{Fore.RED}')
    print(dfName.count(),Fore.WHITE)
    printSeparator()
    print(dfName.head())
    printSeparator()

# %%
import warnings               as     warn
import numpy                  as     np
import pandas                 as     pd
import requests
import datetime

from   bs4                    import BeautifulSoup
from   dateutil.relativedelta import *
from   selenium               import webdriver
from   sklearn.ensemble       import RandomForestClassifier, RandomForestRegressor
from   sklearn.linear_model   import LinearRegression, LogisticRegression
from   sklearn.metrics        import classification_report, precision_score
from   sklearn.neural_network import MLPClassifier
from   sklearn.preprocessing  import StandardScaler

warn.filterwarnings("ignore", category=Warning)
warn.filterwarnings("ignore", category=DeprecationWarning)
warn.filterwarnings("ignore", category=FutureWarning)
warn.filterwarnings("ignore", category=UserWarning)

# %%
start_time00 = datetime.datetime.now()
logStep("ENVIRONMENT PREPARATION")
print(F"Copyright                              : {sys.copyright}")
print(F"OS Platform                            : {sys.platform}")
print(F"OS Name                                : {os.name}")
print(F"OS HOME                                : {os.environ.get('HOME')}")
print(F"OS uName                               : {os.uname().sysname}")
print(F"OS NodeName                            : {os.uname().nodename}")
print(F"OS Release                             : {os.uname().release}")
print(F"OS Release Ver                         : {os.uname().version}")
print(F"OS Machine                             : {os.uname().machine}")
print(F"Process ID                             : {os.getpid()}")
print(F"Parent Process                         : {os.getppid()}")
print(F"OS User                                : {os.getlogin()}")
print(F"OS User ID                             : {os.getuid()}")
print(F"OS Group ID                            : {os.getgid()}")
print(F"OS Effective ID                        : {os.geteuid()}")
print(F"OS Effective GID                       : {os.getegid()}")
print(F"Current dir                            : {os.getcwd()}")
print(F"Python version                         : {sys.version}")
print(F"Version info                           : {sys.version_info}")
print(F"Python API Ver                         : {sys.api_version}")
print(F"Executable                             : {sys.executable}")
print(F"Spark UI                               : http://localhost:4040")
print(F"Spark submit                           : {sys.argv[0]}")
print(F"Hadoop Home                            : {os.environ.get('HADOOP_HOME')}")
print(F"Java Home                              : {os.environ.get('JAVA_HOME')}")
print(F"Current Working Directory              : {os.getcwd()}")
logStep("DONE");
end_time            = datetime.datetime.now()
step00_elapsed_time = end_time - start_time00
logStep(F"ELAPSED TIME: {step00_elapsed_time} seconds")

# %% [markdown]
# Build required data structures - Race DF

# %%
start_time = datetime.datetime.now()
logStep("Build required data structures - Race DF")
races = {'season'   : [], 'round'  : [], 'circuit_id': [], 'lat'    : [],
        'long'      : [], 'country': [], 'date'      : [], 'url'    : []}
url   = 'https://ergast.com/api/f1/{}.json'
for year in list(range(2022,2023)):
    r    = requests.get(url.format(year))
    json = r.json()
    for item in json['MRData']['RaceTable']['Races']:
        try:                   races['season'].append(int(item['season']))
        except Exception as e: races['season'].append(None)
        try:                   races['round'].append(int(item['round']))
        except Exception as e: races['round'].append(None)
        try:                   races['circuit_id'].append(item['Circuit']['circuitId'])
        except Exception as e: races['circuit_id'].append(None)
        try:                   races['lat'].append(float(item['Circuit']['Location']['lat']))
        except Exception as e: races['lat'].append(None)
        try:                   races['long'].append(float(item['Circuit']['Location']['long']))
        except Exception as e: races['long'].append(None)
        try:                   races['country'].append(item['Circuit']['Location']['country'])
        except Exception as e: races['country'].append(None)
        try:                   races['date'].append(item['date'])
        except Exception as e: races['date'].append(None)
        try:                   races['url'].append(item['url'])
        except Exception as e: races['url'].append(None)
races = pd.DataFrame(races)
printDFinfo('races',races)
logStep("DONE");
end_time            = datetime.datetime.now()
step01_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step01_elapsed_time} seconds")

# %% [markdown]
# Build required data structures - Rounds DF

# %%
start_time = datetime.datetime.now()
logStep("Build required data structures - Rounds DF")
rounds = []
i = 0
for year in np.array(races.season.unique()):
    rounds.append([year, list(races[races.season == year]['round'])])
    print(rounds[i][0],rounds[i][1],len(rounds[i][1]))
    i = i + 1
logStep("DONE");
end_time            = datetime.datetime.now()
step02_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step02_elapsed_time} seconds")

# %% [markdown]
# Build required data structures - Race Results DF

# %%
start_time = datetime.datetime.now()
logStep("Build required data structures - Results DF")
results = {'season'       : [], 'round'      : [], 'circuit_id'   : [], 'driver'     : [],
           'date_of_birth': [], 'nationality': [], 'constructor'  : [], 'grid'       : [],
           'time'         : [], 'status'     : [], 'points'       : [], 'podium'     : []}
url     = 'http://ergast.com/api/f1/{}/{}/results.json'
for n in list(range(len(rounds))):
    for i in rounds[n][1]:
        r = requests.get(url.format(rounds[n][0], i))
        json = r.json()
        for item in json['MRData']['RaceTable']['Races'][0]['Results']:
            try:                   results['season'].append(int(json['MRData']['RaceTable']['Races'][0]['season']))
            except Exception as e: results['season'].append(None)
            try:                   results['round'].append(int(json['MRData']['RaceTable']['Races'][0]['round']))
            except Exception as e: results['round'].append(None)
            try:                   results['circuit_id'].append(json['MRData']['RaceTable']['Races'][0]['Circuit']['circuitId'])
            except Exception as e: results['circuit_id'].append(None)
            try:                   results['driver'].append(item['Driver']['driverId'])
            except Exception as e: results['driver'].append(None)
            try:                   results['date_of_birth'].append(item['Driver']['dateOfBirth'])
            except Exception as e: results['date_of_birth'].append(None)
            try:                   results['nationality'].append(item['Driver']['nationality'])
            except Exception as e: results['nationality'].append(None)
            try:                   results['constructor'].append(item['Constructor']['constructorId'])
            except Exception as e: results['constructor'].append(None)
            try:                   results['grid'].append(int(item['grid']))
            except Exception as e: results['grid'].append(None)
            try:                   results['time'].append(int(item['Time']['millis']))
            except Exception as e: results['time'].append(None)
            try:                   results['status'].append(item['status'])
            except Exception as e: results['status'].append(None)
            try:                   results['points'].append(int(item['points']))
            except Exception as e: results['points'].append(None)
            try:                   results['podium'].append(int(item['position']))
            except Exception as e: results['podium'].append(None)
results = pd.DataFrame(results)
printDFinfo('results',results)
logStep("DONE");
end_time            = datetime.datetime.now()
step03_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step03_elapsed_time} seconds")

# %% [markdown]
# Build required data structures - Driver Standings DF

# %%
start_time = datetime.datetime.now()
logStep("Build required data structures - Driver Standings DF")
driver_standings = {'season'     : [], 'round'               : [], 
                    'driver'     : [], 'driver_points'       : [],
                    'driver_wins': [], 'driver_standings_pos': []}
url              = 'https://ergast.com/api/f1/{}/{}/driverStandings.json'
def lookup (df, team, points):
    df['lookup1'] = df.season.astype(str) + df[team] + df['round'].astype(str)
    df['lookup2'] = df.season.astype(str) + df[team] + (df['round']-1).astype(str)
    new_df        = df.merge(df[['lookup1', points]], how = 'left', left_on='lookup2',right_on='lookup1')
    new_df.drop(['lookup1_x', 'lookup2', 'lookup1_y'], axis = 1, inplace = True)
    new_df.rename(columns={f'{points}_x': f'{points}_after_race', f'{points}_y': points}, inplace = True)
    new_df[points].fillna(0, inplace = True)
    return new_df
for n in list(range(len(rounds))):
    for i in rounds[n][1]:
        r = requests.get(url.format(rounds[n][0], i))
        json = r.json()
        for item in json['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']:
            try:                   driver_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
            except Exception as e: driver_standings['season'].append(None)
            try:                   driver_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
            except Exception as e: driver_standings['round'].append(None)
            try:                   driver_standings['driver'].append(item['Driver']['driverId'])
            except Exception as e: driver_standings['driver'].append(None)
            try:                   driver_standings['driver_points'].append(int(item['points']))
            except Exception as e: driver_standings['driver_points'].append(None)
            try:                   driver_standings['driver_wins'].append(int(item['wins']))
            except Exception as e: driver_standings['driver_wins'].append(None)
            try:                   driver_standings['driver_standings_pos'].append(int(item['position']))
            except Exception as e: driver_standings['driver_standings_pos'].append(None)
driver_standings = pd.DataFrame(driver_standings)
driver_standings = lookup(driver_standings, 'driver', 'driver_points')
driver_standings = lookup(driver_standings, 'driver', 'driver_wins')
driver_standings = lookup(driver_standings, 'driver', 'driver_standings_pos')
driver_standings.drop(['driver_points_after_race', 'driver_wins_after_race', 'driver_standings_pos_after_race'], axis    = 1, inplace = True)
printDFinfo('driver_standings',driver_standings)
logStep("DONE");
end_time            = datetime.datetime.now()
step04_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step04_elapsed_time} seconds")

# %% [markdown]
# Build required data structures - Constructor DF

# %%
start_time = datetime.datetime.now()
logStep("Build required data structures - Constructor DF")
constructor_rounds    = rounds
constructor_standings = {'season'          : [], 'round'                    : [],
                         'constructor'     : [], 'constructor_points'       : [],
                         'constructor_wins': [], 'constructor_standings_pos': []}
url                   = 'https://ergast.com/api/f1/{}/{}/constructorStandings.json'
for n in list(range(len(constructor_rounds))):
    for i in constructor_rounds[n][1]:
        r    = requests.get(url.format(constructor_rounds[n][0], i))
        json = r.json()
        for item in json['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']:
            try:                   constructor_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
            except Exception as e: constructor_standings['season'].append(None)
            try:                   constructor_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
            except Exception as e: constructor_standings['round'].append(None)
            try:                   constructor_standings['constructor'].append(item['Constructor']['constructorId'])
            except Exception as e: constructor_standings['constructor'].append(None)
            try:                   constructor_standings['constructor_points'].append(int(item['points']))
            except Exception as e: constructor_standings['constructor_points'].append(None)
            try:                   constructor_standings['constructor_wins'].append(int(item['wins']))
            except Exception as e: constructor_standings['constructor_wins'].append(None)
            try:                   constructor_standings['constructor_standings_pos'].append(int(item['position']))
            except Exception as e: constructor_standings['constructor_standings_pos'].append(None)
constructor_standings = pd.DataFrame(constructor_standings)
constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_points')
constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_wins')
constructor_standings = lookup(constructor_standings, 'constructor', 'constructor_standings_pos')
constructor_standings.drop(['constructor_points_after_race','constructor_wins_after_race','constructor_standings_pos_after_race'],axis=1,inplace=True)
printDFinfo('Constructor_Standings',constructor_standings)
logStep("DONE");
end_time            = datetime.datetime.now()
step05_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step05_elapsed_time} seconds")

# %% [markdown]
# Build required data structures - Qualifying DF

# %%
start_time = datetime.datetime.now()
logStep("Build required data structures - Qualifying DF")
qualifying_results = pd.DataFrame()
for year in list(range(2022,2023)):
    url = 'https://www.formula1.com/en/results.html/{}/races.html'
    r = requests.get(url.format(year))
    soup = BeautifulSoup(r.text, 'html.parser')
    # find links to all circuits for a certain year
    year_links = []
    for page in soup.find_all('a', attrs = {'class':"resultsarchive-filter-item-link FilterTrigger"}):
        link = page.get('href')
        if f'/en/results.html/{year}/races/' in link: 
            year_links.append(link)
    # for each circuit, switch to the starting grid page and read table
    year_df = pd.DataFrame()
    new_url = 'https://www.formula1.com{}'
    for n, link in list(enumerate(year_links)):
        link = link.replace('race-result.html', 'starting-grid.html')
        df = pd.read_html(new_url.format(link))
        df = df[0]
        df['season'] = year
        df['round'] = n+1
        for col in df:
            if 'Unnamed' in col:
                df.drop(col, axis = 1, inplace = True)
        year_df = pd.concat([year_df, df])
    # concatenate all tables from all years  
    qualifying_results = pd.concat([qualifying_results, year_df])
qualifying_results.rename(columns = {'Pos': 'grid', 'Driver': 'driver_name', 'Car': 'car', 'Time': 'qualifying_time'}, inplace = True)
qualifying_results.drop('No', axis = 1, inplace = True)
qualifying_results.qualifying_time = qualifying_results.grid
printDFinfo('Qualifying_Results',qualifying_results)
logStep("DONE");
end_time            = datetime.datetime.now()
step06_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step06_elapsed_time} seconds")

# %% [markdown]
# Build required data structures - Weather DF

# %%
start_time = datetime.datetime.now()
logStep("Build required data structures - Weather DF")
weather = races.iloc[:,[0,1,2]]
info    = []
# read wikipedia tables
for link in races.url:
    try:
        df = pd.read_html(link)[0]
        if 'Weather' in list(df.iloc[:,0]):
            n = list(df.iloc[:,0]).index('Weather')
            info.append(df.iloc[n,1])
        else:
            df = pd.read_html(link)[1]
            if 'Weather' in list(df.iloc[:,0]):
                n = list(df.iloc[:,0]).index('Weather')
                info.append(df.iloc[n,1])
            else:
                df = pd.read_html(link)[2]
                if 'Weather' in list(df.iloc[:,0]):
                    n = list(df.iloc[:,0]).index('Weather')
                    info.append(df.iloc[n,1])
                else:
                    df = pd.read_html(link)[3]
                    if 'Weather' in list(df.iloc[:,0]):
                        n = list(df.iloc[:,0]).index('Weather')
                        info.append(df.iloc[n,1])
                    else:
                        driver = webdriver.Chrome()
                        driver.get(link)
                        # click language button
                        button = driver.find_element_by_link_text('Italiano')
                        button.click()
                        # find weather in italian with selenium
                        clima = driver.find_element_by_xpath('//*[@id="mw-content-text"]/div/table[1]/tbody/tr[9]/td').text
                        info.append(clima)  
    except Exception as e:
        info.append('not found')
# append column with weather information to dataframe  
weather['weather'] = info
# set up a dictionary to convert weather information into keywords
weather_dict = {'weather_warm' : ['soleggiato', 'clear'        , 
                                  'warm'      , 'hot'          , 
                                  'sunny'     , 'fine'         , 
                                  'mild'      , 'sereno'      ],
               'weather_cold'  : ['cold'      , 'fresh'        , 
                                  'chilly'    , 'cool'        ],
               'weather_dry'   : ['dry'       , 'asciutto'    ],
               'weather_wet'   : ['showers'   , 'wet'          ,  
                                  'rain'      , 'pioggia'      , 
                                  'damp'      , 'thunderstorms', 
                                  'rainy'                     ],
               'weather_cloudy': ['overcast'  , 'nuvoloso'     , 
                                  'clouds'    , 'cloudy'       , 
                                  'grey'      , 'coperto']}
# map new df according to weather dictionary
weather_df = pd.DataFrame(columns = weather_dict.keys())
for col in weather_df:
    weather_df[col] = weather['weather'].map(lambda x: 1 if any(i in weather_dict[col] for i in x.lower().split()) else 0)
weather_info = pd.concat([weather, weather_df], axis = 1)
printDFinfo('weather_info',weather_info)
logStep("DONE");
end_time            = datetime.datetime.now()
step07_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step07_elapsed_time} seconds")

# %% [markdown]
# Merge all DFs into one single DF to be used by ML training and prediction

# %%
start_time = datetime.datetime.now()
logStep("Build required data structures - Merge DFs")
df1      = pd.merge(races, weather,               how='inner', on=['season', 'round', 'circuit_id']).drop(['lat', 'long','country','weather'],axis = 1)
df2      = pd.merge(df1,   results,               how='inner', on=['season', 'round', 'circuit_id']).drop(['points', 'status', 'time'],axis = 1)
df3      = pd.merge(df2,   driver_standings,      how='left',  on=['season', 'round', 'driver'])
df4      = pd.merge(df3,   constructor_standings, how='left',  on=['season', 'round', 'constructor'])
final_df = pd.merge(df4,   qualifying_results,    how='inner', on=['season', 'round', 'grid']).drop(['driver_name', 'car'],axis = 1)
final_df['date']          = pd.to_datetime(final_df.date)
final_df['date_of_birth'] = pd.to_datetime(final_df.date_of_birth)
final_df['driver_age']    = final_df.apply(lambda x: relativedelta(x['date'], x['date_of_birth']).years, axis=1)
final_df.drop(['date', 'date_of_birth'], axis = 1, inplace = True)
for col in ['driver_points', 'driver_wins', 'driver_standings_pos', 'constructor_points', 'constructor_wins' , 'constructor_standings_pos']:
    final_df[col].fillna(0, inplace = True)
    final_df[col] = final_df[col].map(lambda x: int(x))
final_df.dropna(inplace = True )
final_df = final_df[final_df['qualifying_time'] != 0]
final_df.sort_values(['season', 'round', 'grid'], inplace = True)
final_df['qualifying_time_diff'] = final_df.groupby(['season', 'round']).qualifying_time.diff()
final_df['qualifying_time'] = final_df.groupby(['season','round']).qualifying_time_diff.cumsum().fillna(0)
final_df.drop('qualifying_time_diff', axis = 1, inplace = True)
final_df.reset_index(inplace = True, drop = True)
printDFinfo('final_df',final_df)
logStep("DONE");
end_time            = datetime.datetime.now()
step08_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step08_elapsed_time} seconds")

# %% [markdown]
# Prepare score functions - Regression and Classification

# %%
start_time = datetime.datetime.now()
logStep("Prepare score functions - Regression and Classification")
scaler    = StandardScaler()
def score_regression(model):
    score = 0
    for circuit in df['round'].unique():
        test = df[(df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis = 1)
        y_test = test.podium
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
        prediction_df = pd.DataFrame(model.predict(X_test), columns = ['results'])
        prediction_df['podium'] = y_test.reset_index(drop = True)
        prediction_df['actual'] = prediction_df.podium.map(lambda x: 1 if x == 1 else 0)
        prediction_df.sort_values('results', ascending = True, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted']        = prediction_df.index
        prediction_df['predicted']        = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)
        score += precision_score(prediction_df.actual, prediction_df.predicted)
    model_score = score / df['round'].unique().max()
    return model_score, prediction_df

def score_classification(model):
    score = 0
    for circuit in df['round'].unique():
        test = df[(df['round'] == circuit)]
        X_test = test.drop(['driver', 'podium'], axis = 1)
        y_test = test.podium
        X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
        prediction_df = pd.DataFrame(model.predict_proba(X_test), columns = ['proba_0', 'proba_1'])
        prediction_df['actual'] = y_test.reset_index(drop = True)
        prediction_df.sort_values('proba_1', ascending = False, inplace = True)
        prediction_df.reset_index(inplace = True, drop = True)
        prediction_df['predicted'] = prediction_df.index
        prediction_df['predicted'] = prediction_df.predicted.map(lambda x: 1 if x == 0 else 0)
        score += precision_score(prediction_df.actual, prediction_df.predicted)
    model_score = score / df['round'].unique().max()
    return model_score, prediction_df
    
logStep("DONE");
end_time            = datetime.datetime.now()
step09_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step09_elapsed_time} seconds")

# %% [markdown]
# Scaling

# %%
start_time = datetime.datetime.now()
logStep("Scaling and Splitting")
df        = final_df.copy()
df.reset_index(inplace = True, drop = True)
df        = df.drop(['circuit_id'] , axis = 1)
df        = df.drop(['url']        , axis = 1)
df        = df.drop(['nationality'], axis = 1)
df        = df.drop(['constructor'], axis = 1)
df.podium = df.podium.map(lambda x: 1 if x == 1 else 0)
train     = df[df.season <2023]
X_train   = train.drop(['driver', 'podium'], axis = 1)
y_train   = train.podium
X_train   = pd.DataFrame(scaler.fit_transform(X_train), columns = X_train.columns)
logStep("DONE");
end_time            = datetime.datetime.now()
step10_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step10_elapsed_time} seconds")

# %% [markdown]
# Linear Regression

# %%
start_time = datetime.datetime.now()
logStep("Linear Regression")
comparison_dict = {'model': [], 'params': [], 'score': []}
params={'fit_intercept': [True, False]}
for fit_intercept in params['fit_intercept']:
    model_params = (fit_intercept)
    model = LinearRegression(fit_intercept = fit_intercept)
    model.fit(X_train, y_train)   
    model_score, model_prediction = score_regression(model)
    comparison_dict['model'].append('linear_regression')
    comparison_dict['params'].append(model_params)
    comparison_dict['score'].append(model_score)
print(pd.DataFrame(comparison_dict).groupby('model')['score'].max())
predictions  = model.predict(X_train)
target_names = df.driver.unique()
newpred      = np.zeros(len(predictions))
for i in range(len(predictions)):
  newpred[i] = predictions[i]
p  = np.nan_to_num(newpred)
j = 0
for i in p:
    i = i * 100
    i = int(i)
    p[j] = i
    j = j + 1
y  = np.nan_to_num(y_train) 
print(classification_report(y, p, target_names = target_names,labels=np.unique(p)))
logStep("DONE");
end_time            = datetime.datetime.now()
step11_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step11_elapsed_time} seconds")

# %% [markdown]
# Random Forest Regressor

# %%
start_time = datetime.datetime.now()
logStep("Random Forest Regressor")
params={'criterion': ['friedman_mse'],
        'max_features': [0.8, 1, None],
        'max_depth': [None]}
for criterion in params['criterion']:
    for max_features in params['max_features']:
        for max_depth in params['max_depth']:
            model_params = (criterion, max_features, max_depth)
            model = RandomForestRegressor(criterion    = criterion,
                                          max_features = max_features, 
                                          max_depth    = max_depth, 
                                          random_state = 1)
            model.fit(X_train, y_train)
            model_score, prediction_df = score_regression(model)
            comparison_dict['model'].append('random_forest_regressor')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)
print(pd.DataFrame(comparison_dict).groupby('model')['score'].max())
predictions = model.predict(X_train)
target_names = df.driver.unique()
newpred = np.zeros(len(predictions))
for i in range(len(predictions)):
    newpred[i] = predictions[i]
p  = np.nan_to_num(newpred)
j = 0
for i in p:
    i = i * 100
    i = int(i)
    p[j] = i
    j = j + 1
y  = np.nan_to_num(y_train) 
print(classification_report(y, p, target_names = target_names,labels=np.unique(p)))
logStep("DONE");
end_time            = datetime.datetime.now()
step12_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step12_elapsed_time} seconds")

# %% [markdown]
# Logistic Regression

# %%
start_time = datetime.datetime.now()
logStep("Logistic Regression")
params={'penalty': ['l1', 'l2'],
        'solver' : ['saga', 'liblinear'],
        'C'      : np.logspace(-3,1,20)}
for penalty in params['penalty']:
    for solver in params['solver']:
        for c in params['C']:
            model_params = (penalty, solver, c)
            model = LogisticRegression(penalty = penalty, solver = solver, C = c, max_iter = 10000)
            model.fit(X_train, y_train)
            model_score, model_prediction = score_classification(model)
            comparison_dict['model'].append('logistic_regression')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)
print(pd.DataFrame(comparison_dict).groupby('model')['score'].max())
predictions = model.predict(X_train)
target_names = df.driver.unique()
# Print the predictions
newpred = np.zeros(len(predictions))
for i in range(len(predictions)):
    newpred[i] = predictions[i]
p  = np.nan_to_num(newpred)
j = 0
for i in p:
    i = i * 100
    i = int(i)
    p[j] = i
    j = j + 1
y  = np.nan_to_num(y_train) 
print(classification_report(y, p, target_names = target_names,labels=np.unique(p)))
logStep("DONE");
end_time            = datetime.datetime.now()
step13_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step13_elapsed_time} seconds")

# %% [markdown]
# Random Forest Classifier

# %%
start_time = datetime.datetime.now()
logStep("Random Forest Classifier")
params = {
    'criterion'   : ['gini', 'entropy'],
    'max_features': [None],
    'max_depth'   : [None]
    }
for criterion in params['criterion']:
    for max_features in params['max_features']:
        for max_depth in params['max_depth']:
            model_params = (criterion, max_features, max_depth)
            model = RandomForestClassifier(criterion = criterion, max_features = max_features, max_depth = max_depth)
            model.fit(X_train, y_train)
            model_score, model_prediction = score_classification(model)
            comparison_dict['model'].append('random_forest_classifier')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score)
print(pd.DataFrame(comparison_dict).groupby('model')['score'].max())
predictions  = model.predict(X_train)
target_names = df.driver.unique()
newpred = np.zeros(len(predictions))
for i in range(len(predictions)):
    newpred[i] = predictions[i]
p  = np.nan_to_num(newpred)
j = 0
for i in p:
    i = i * 100
    i = int(i)
    p[j] = i
    j = j + 1
y  = np.nan_to_num(y_train) 
print(classification_report(y, p, target_names = target_names,labels=np.unique(p)))
logStep("DONE");
end_time            = datetime.datetime.now()
step14_elapsed_time = end_time - start_time
logStep(F"ELAPSED TIME: {step14_elapsed_time} seconds")

# %% [markdown]
# Neural Network

# %%
start_time = datetime.datetime.now()
logStep("Neural Network Classifier")
params={'hidden_layer_sizes': [(60,20,40,5), (50,25,50,10)], 
        'activation'        : ['tanh', 'relu'], 
        'solver'            : 'adam', 
        'alpha'             : np.logspace(-4,2,20)} 
for hidden_layer_sizes in params['hidden_layer_sizes']:
    for activation in params['activation']:
        for alpha in params['alpha']:
            model_params = (hidden_layer_sizes, activation, 'adam', alpha )
            model = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = 'adam', alpha = alpha, random_state = 1)
            model.fit(X_train, y_train)
            model_score, model_prediction = score_classification(model)
            comparison_dict['model'].append('neural_network_classifier')
            comparison_dict['params'].append(model_params)
            comparison_dict['score'].append(model_score) 
print(pd.DataFrame(comparison_dict).groupby('model')['score'].max())
predictions = model.predict(X_train)
target_names = df.driver.unique()
newpred = np.zeros(len(predictions))
for i in range(len(predictions)):
    newpred[i] = predictions[i]
p  = np.nan_to_num(newpred)
j = 0
for i in p:
    i = i * 100
    i = int(i)
    p[j] = i
    j = j + 1
y  = np.nan_to_num(y_train) 
print(classification_report(y, p, target_names = target_names,labels=np.unique(p)))
logStep("DONE");
end_time            = datetime.datetime.now()
step15_elapsed_time = end_time - start_time

end_time_tt         = datetime.datetime.now()
steptt_elapsed_time = end_time_tt - start_time00
logStep(F"ELAPSED TIME: {step15_elapsed_time} seconds")
logStep(F"TOT ELA TIME: {steptt_elapsed_time} seconds")


