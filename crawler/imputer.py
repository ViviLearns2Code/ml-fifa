import os
import csv


def setZeroValue(inputfile, dict_min):
    ''' fills in missing (zero) market values 
    with the minimum market value within the team '''

    # output writer
    head, filename = os.path.split(inputfile)
    dir = os.path.dirname(__file__)
    outfilepath = os.path.abspath(os.path.join(dir, 'output/imputed/{filename}')).format(filename=filename)
    outfile = open(outfilepath, 'a', encoding='utf-8')
    writer = csv.writer(outfile, delimiter=';', lineterminator='\n')
    
    # input reader
    f1 = open(inputfile, newline='\n', encoding='utf-8')
    reader1 = csv.reader(f1, delimiter=";")

    for iRow in reader1:
        gameid = iRow[0]
        team = iRow[1]
        club = iRow[2]
        name = iRow[3]
        age = iRow[4]
        position = iRow[5]
        value = iRow[6]
        start = iRow[7]

        if value != "value": # not header
          value = float(value)
          if value == float(0.0):
            value = dict_min[team]

        writer.writerow([gameid, team, club, name,
                         age, position, value, start])

    outfile.close()
    f1.close()


def findAndSetMinNonZeroVaule(fileName):
    # dictionary: key = team name, value = smallest non-zero value of team
    dict_min = {} 

    f = open(fileName, newline='\n', encoding='utf-8')
    reader = csv.DictReader(f, delimiter=";")
    
    existNonZeroValue = False

    for row in reader:
        value = row["value"]
        value = float(value)
        team = row["team"]

        if team not in dict_min and value != float(0.0): # initialize team minima
          dict_min[team] = value
        elif value != float(0.0): # team is already being recorded
            if value < dict_min[team]:
                dict_min[team] = value
        elif value == float(0.0): # there is at least one value missing, imputation necessary
            existNonZeroValue = True
    f.close()
    if existNonZeroValue == True:
        setZeroValue(fileName, dict_min)

dir = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(dir, 'output/matches'))

for filename in os.listdir(path):
    inputpath = os.path.abspath(os.path.join(dir, 'output/matches/{file}')).format(file=filename)
    print(inputpath)
    findAndSetMinNonZeroVaule(inputpath)