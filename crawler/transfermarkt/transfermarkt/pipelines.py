# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import csv
import os

dir = os.path.dirname(__file__)
filepath_gamelist = os.path.abspath(os.path.join(dir, '../../output/{name}_gamelist.csv'))
filepath_match = os.path.abspath(os.path.join(dir, '../../output/matches/{id}.csv'))
filepath_compare = os.path.abspath(os.path.join(dir, '../../output/compare/{idA}_{idB}.csv'))

class GamelistPipeline(object):
    def process_item(self, item, spider):
        if spider.name == 'gamelist':
            filename = filepath_gamelist.format(name=item["tournament"])
            file_exists = os.path.isfile(filename)
            if file_exists:
                file_empty = os.stat(filename).st_size == 0
            else:
                file_empty = True
            headers = ['tournament', 'gametype', 'teamA', 'teamidA', 'teamB', 'teamidB', 'resultA', 'resultB', 'addinfo', 'gameid','date']
            with open(filename, 'a', encoding='utf-8') as f:
                writer = csv.DictWriter(f, delimiter=';', lineterminator='\n',fieldnames=headers)
                if file_empty:
                    writer.writeheader()
                writer.writerow(item)
        if spider.name == 'match':
            filename = filepath_match.format(id=item["gameid"])
            file_exists = os.path.isfile(filename)
            if file_exists:
                file_empty = os.stat(filename).st_size == 0
            else:
                file_empty = True
            headers = ['gameid', 'team', 'club', 'name', 'age', 'position', 'value', 'start']
            with open(filename, 'a', encoding='utf-8') as f:
                writer = csv.DictWriter(f,delimiter=';',lineterminator='\n',fieldnames=headers)
                if file_empty:
                    writer.writeheader()
                writer.writerow(item)
        if spider.name == 'comparison':
            filename1 = filepath_compare.format(idA=item["teamidA"],idB=item["teamidB"])
            filename2 = filepath_compare.format(idA=item["teamidB"],idB=item["teamidA"])
            if os.path.isfile(filename1) == True:
                filename = filename1
                file_exists = True
            elif os.path.isfile(filename2) == True:
                filename = filename2
                file_exists = True
            else:
                filename = filename1
                file_exists = False
                # create file in next step
            if file_exists:
                file_empty = os.stat(filename).st_size == 0
            else:
                file_empty = True               
            headers = ['gametype', 'gamecontext', 'gamedate', 'teamidA', 'teamidB', 'resultA', 'resultB','for_game']
            with open(filename, 'a+', encoding='utf-8') as f:
                # for_game field just a helper to record if past encounters of two teams habe already been recorded
                reader = csv.DictReader(f,delimiter=';',lineterminator='\n',fieldnames=headers)
                if not file_empty:
                    f.seek(0)
                    next(reader)
                    first_item = next(reader)
                    print(first_item["for_game"],item["for_game"])
                    if first_item["for_game"]!=item["for_game"]:
                        return
                writer = csv.DictWriter(f,delimiter=';',lineterminator='\n',fieldnames=headers)
                if file_empty:
                    writer.writeheader()
                
                writer.writerow(item)
        return item

