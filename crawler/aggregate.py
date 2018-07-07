import os
import csv
from datetime import datetime
from datetime import timedelta
import pandas as pd
import numpy as np

position_category = {
    "Torwart": "defence",
    "Innenverteidiger": "defence",
    "Linker Verteidiger": "defence",
    "Rechter Verteidiger": "defence",
    "Defensives Mittelfeld": "defence",
    "Zentrales Mittelfeld": "offence",  # we count central midfield as offence
    "Offensives Mittelfeld": "offence",
    "Linkes Mittelfeld": "offence",
    "Rechtes Mittelfeld": "offence",
    "Rechtsaußen": "offence",
    "Linksaußen": "offence",
    "Hängende Spitze": "offence",
    "Mittelstürmer": "offence",
}

country_names = {
    "Demokratische-Republik-Kongo": "Demokratische Republik Kongo",
    "Kongo": "Republik Kongo",
    "Vereinigte-Arabische-Emirate": "Vereinigte Arabische Emirate",
    "USA": "Vereinigte Staaten",
    "Trinidad": "Trinidad und Tobago",
    "Serbien-Montenegro": "Serbien und Montenegro"
}

def useful(game):
    ''' Exclude games with corrupt data '''
    if game["gameid"] == "1027049" or game["gameid"] == "986801":
        # useless/incomplete data
        return False
    else:
        return True

def add_items(file, items=[]):
    ''' Add all items from file which can be used for later analysis '''
    with open(file=file, mode='r', encoding='utf-8') as f:
        # https://docs.python.org/3/library/csv.html
        # If fieldnames is omitted, the values in the first row of file f will be used as the fieldnames.
        # Regardless of how the fieldnames are determined, the ordered dictionary preserves their original ordering.
        reader = csv.DictReader(f=f, delimiter=";")
        for row in reader:
            if useful(row):
                items.append(row)
        return items

def is_after(check_date,target_date,target_format):
    ''' Checks if check_date is after target_date '''
    #https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    check = datetime.strptime(check_date,check_format)
    target = datetime.strptime(target_date,target_format)
    return check >= target

def years_between(dateA,formatA,dateB,formatB):
    ''' Checks if check_date is after target_date '''
    dateA = datetime.strptime(dateA,formatA)
    dateB = datetime.strptime(dateB,formatB)
    return abs(dateA-dateB)/timedelta(days=1)*1/365.2425

def calc_mva(alpha,series):
    ''' Calculate moving average
    alpha: weight of previous values
    series: array with values from most recent to oldest '''
    result = 0
    for i in reversed(series):
        result = alpha*result+(1-alpha)*i
    return result

dir = os.path.dirname(__file__)

# Stage 1: Get all games for all tournaments
path = os.path.abspath(os.path.join(dir, 'output/'))
games = []
for filename in os.listdir(path):
    if os.path.isfile(os.path.join(path, filename)):
        tournamentfile = os.path.abspath(os.path.join(
            dir, 'output/{file}')).format(file=filename)
        games = add_items(tournamentfile, games)

# Stage 2: Aggregate team stats
for game in games:
    try:
        gameid = game["gameid"]
        teamA = game["teamA"] if game["teamA"] not in country_names else country_names[game["teamA"]]
        teamB = game["teamB"] if game["teamB"] not in country_names else country_names[game["teamB"]]

        gamefile = os.path.abspath(os.path.join(
            dir, 'output/matches/{gameid}.csv'.format(gameid=gameid)))
        # open match lineup file
        lineup = pd.read_csv(filepath_or_buffer=gamefile,
                             sep=";", index_col=0, low_memory=False)
        # special case: deceased players have a † sign in front of their age 
        temp = lineup["age"].astype("str").str.replace("†","")                
        lineup["age"] = temp.astype("int")
        # group by team name to calculate avg. age
        mean_age = lineup[["team", "age"]].groupby(["team"]).mean()
        # group by team and position to calculate mv of defense/attack
        sum_value = {}
        team_frag = {}
        team_group = lineup[["team", "value", "position", "club"]].groupby([
            "team"])
        for group in team_group:
            sum_value[group[0]] = group[1].set_index(
                "position").groupby(position_category).sum()
            team_frag[group[0]] = group[1]["club"].nunique(
                dropna=False)  # count missing values as own club

        game["teamA_age"] = mean_age["age"][teamA]
        game["teamB_age"] = mean_age["age"][teamB]
        game["teamA_def_val"] = sum_value[teamA]["value"]["defence"]
        game["teamB_def_val"] = sum_value[teamB]["value"]["defence"]
        game["teamA_off_val"] = sum_value[teamA]["value"]["offence"]
        game["teamB_off_val"] = sum_value[teamB]["value"]["offence"]
        game["teamA_frag"] = team_frag[teamA]
        game["teamB_frag"] = team_frag[teamB]

    except KeyError:
        print("Gameid-", gameid, "-Team A-",
              game["teamA"], "-Team B-", game["teamB"])
        #print("Average Age per Team:", mean_age)
        #print("Average Value per Team Position:", mean_value)
        #print("Fragmentation per Team:", team_frag)
        raise Exception

# Stage 3: Aggregate past encounters
filepath_compare = os.path.abspath(os.path.join(dir, 'output/compare/{idA}_{idB}.csv'))
comp_dtype = {
    "gametype": str,
    "gamecontext": str,
    "gamedate": str,
    "teamidA": str,
    "teamidB": str,
    "resultA": str,
    "resultB": str
}

for game in games:
    filename1 = filepath_compare.format(idA=game["teamidA"],idB=game["teamidB"])
    filename2 = filepath_compare.format(idA=game["teamidB"],idB=game["teamidA"])
    if os.path.isfile(filename1) == True:
        compfile = filename1
    elif os.path.isfile(filename2) == True:
        compfile = filename2
    print(compfile)
    # open comparison file
    df_comp = pd.read_csv(filepath_or_buffer=compfile, sep=";", low_memory=False, dtype=comp_dtype)
    # drop all encounters with missing results
    df_comp.drop(df_comp[df_comp["resultA"]=="-"].index,inplace=True)
    df_comp[["resultA","resultB"]] = df_comp[["resultA","resultB"]].astype(np.float64)
    # drop all encounters which took place after the tournament game
    df_comp["temp"] = df_comp["gamedate"].apply(
        lambda x: is_after(x,"%d.%m.%Y",game["date"],"%d.%m.%y") 
        or years_between(x,"%d.%m.%Y",game["date"],"%d.%m.%y")>15) 
    df_comp.drop(df_comp[df_comp["temp"]==True].index,inplace=True) 
    if df_comp.empty:
        # write N/A
        game["past_resultA"] = ""
        game["past_resultB"] = ""
        continue
    # calculate weighted moving average of goals scored in past games
    # pitfall: teamA in df_comp is not always equal to teamA in game
    # => use ids  
    alpha = 0.05
    teamApast = []
    teamBpast = []
    for i,encounter in df_comp.iterrows():
        if encounter["teamidA"] == game["teamidA"]:
            teamApast.append(encounter["resultA"])
            teamBpast.append(encounter["resultB"])
        elif encounter["teamidB"] == game["teamidA"]:
            teamApast.append(encounter["resultB"])
            teamBpast.append(encounter["resultA"])
    game["past_resultA"] = calc_mva(alpha,teamApast)
    game["past_resultB"] = calc_mva(alpha,teamBpast)        

# Stage 4: Write to final csv
with open('final.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["gameid", "tournament", "gametype", "teamA", "teamidA", "teamB", "teamidB", "resultA", "resultB", "addinfo", "date",
                  "teamA_age", "teamB_age", "teamA_def_val", "teamB_def_val", "teamA_off_val", "teamB_off_val", 
                  "teamA_frag", "teamB_frag", "past_resultA", "past_resultB"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
    writer.writeheader()
    writer.writerows(games)
