import scrapy
from .. import items
import os
import csv

dir = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(dir, '../../../output/'))
unit_abbreviations = [
    { "abbr": "Mio.", "factor": "1000000"},
    { "abbr": "Tsd.", "factor": "1000"}
]

def conv2number(decimal, unit):
    for u in unit_abbreviations:
        if u['abbr'] == unit:
            factor = u['factor']
            total = float(decimal.replace(',','.'))*int(factor)
            return total

class MatchScraper(scrapy.Spider):
    '''
    This scraper gathers the lineup data of a game
    '''
    name = "match"

    def start_requests(self):
        urls = []
        for filename in os.listdir(path):
            filepath = os.path.join(path,filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f,delimiter=';')
                    next(reader) # skip header
                    for row in reader:
                        teamA = row[2]
                        teamB = row[4]
                        id = row[9]
                        url = 'https://www.transfermarkt.de/{A}_{B}/aufstellung/spielbericht/{id}'.format(A=teamA,B=teamB,id=id)
                        urls.append(url)
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
    def parse(self, response):
        if response.status == 404:
            # stop crawling so error can be fixed
            print("ERROR for {url}".format(url=response.url))
            raise CloseSpider(reason='Could not access URL')
        gameid = response.url.split('/')[-1]
        for box in response.css('.row.sb-formation .large-6.columns .box')[0:4]:
            # Starting 11/Sub
            role = box.css('.table-header::text').extract_first()
            team = box.xpath('.//img/@alt').extract_first()
            for row in box.css('.responsive-table .items').xpath('tr'):                
                cells = row.xpath('td')
                position = cells[0].xpath('@title').extract_first()
                club = cells[3].xpath('.//img/@alt').extract_first()
                subrows = cells[1].xpath('table/tr')
                name = subrows[0].xpath('td/a/text()').extract()[2]
                value = subrows[1].xpath('td/text()').extract_first()
                value = value.split(",",1)[1]
                if value == '-' or value == ' -':
                    value = 0 # no data available, set to zero
                else:
                    value_raw = value.strip().split(" ")
                    value = conv2number(value_raw[0], value_raw[1])
                age = subrows[0].xpath('td/text()').extract()[2]
                age = age.replace('\r\n', '').strip().split('(',1)[1].split(')',1)[0]
                age = age.replace('Jahre', '').strip()
                
                yield items.LineupItem(
                    gameid = gameid,
                    name = name,
                    age = age,
                    value = value,
                    club = club,
                    team = team,
                    position = position,
                    start = role
                )
