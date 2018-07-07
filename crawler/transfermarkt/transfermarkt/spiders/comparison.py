import scrapy
from .. import items
import os
import csv

dir = os.path.dirname(__file__)
path = os.path.abspath(os.path.join(dir, '../../../output/'))

class ComparisonScraper(scrapy.Spider):
    '''
    This scraper gathers the team comparisons (results of past games between two teams)
    '''
    name = "comparison"

    def start_requests(self):
        urls = []
        gameids = {}
        for filename in os.listdir(path):
            filepath = os.path.join(path,filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f,delimiter=';')
                    next(reader) # skip header
                    for row in reader:
                        teamidA = row[3]
                        teamidB = row[5]
                        url = 'https://www.transfermarkt.de/vergleich/bilanzdetail/verein/{idA}/gegner_id/{idB}'.format(idA=teamidA,idB=teamidB)
                        urls.append(url)
                        gameids[url]=row[9]
        for url in urls:
            yield scrapy.Request(url=url,callback=self.parse,meta={"for_game":gameids[url]})
    def parse(self, response):
        if response.status == 404:
            # stop crawling so error can be fixed
            print("ERROR for {url}".format(url=response.url))
            raise CloseSpider(reason='Could not access URL')
        table = response.css('.responsive-table .items')[0]
        for row in table.xpath('.//tbody/tr'):
            # td in each row: index 1 (type of game), index 2 (ctx of game), index 4 (date), index 6-8 (result)
            cells = row.xpath('td')
            gametype = cells[1].xpath('.//td[contains(@class,"hauptlink")]/a/@title').extract_first()
            gamectx = cells[2].xpath('.//a/text()').extract_first()
            gamedate = cells[4].xpath('.//text()').extract_first()
            teamidA = cells[6].xpath('.//a/@id').extract_first()
            results = cells[7].xpath('.//a/span/text()').extract_first().split(':')
            teamidB = cells[8].xpath('.//a/@id').extract_first()
            resultA = results[0]
            resultB = results[1]

            yield items.CompareItem(
                for_game = response.meta["for_game"],
                gametype = gametype,
                gamecontext = gamectx,
                gamedate = gamedate,
                teamidA = teamidA,
                teamidB = teamidB,
                resultA = resultA,
                resultB = resultB
            )