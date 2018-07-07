# https://doc.scrapy.org/en/latest/intro/tutorial.html
# https://stackoverflow.com/questions/43468275/running-scrapy-from-a-script-with-file-output
import scrapy
from scrapy.crawler import CrawlerProcess
from .. import items
import re

country_abbreviations = [
    { "abbr": "Bosnien-H.", "full": "Bosnien-Herzegowina"},
    { "abbr": "Serbien-Mont.", "full": "Serbien-Montenegro"},
    { "abbr": "V. A. E.", "full": "Vereinigte-Arabische-Emirate"},
    { "abbr": "DR Kongo", "full": "Demokratische-Republik-Kongo"},
    { "abbr": "Äquat.guinea", "full": "Äquatorialguinea"}
]
def search(country):
    for c in country_abbreviations:
        if c['abbr'] == country:
            return c['full']
    return country

class GameListScraper(scrapy.Spider):
    '''
    This scraper gathers all games of a tournament
    '''
    name = "gamelist"
    
    def start_requests(self):
        urls = [
            # World Cup
            'https://www.transfermarkt.de/weltmeisterschaft-2010/gesamtspielplan/pokalwettbewerb/WM10/saison_id/2009',
            'https://www.transfermarkt.de/weltmeisterschaft-2006/gesamtspielplan/pokalwettbewerb/WM06/saison_id/2005',
            'https://www.transfermarkt.de/weltmeisterschaft-2014/gesamtspielplan/pokalwettbewerb/WM14/saison_id/2013',
            # European Championsips
            'https://www.transfermarkt.de/europameisterschaft-2008/gesamtspielplan/pokalwettbewerb/EM08/saison_id/2007',
            'https://www.transfermarkt.de/europameisterschaft-2012/gesamtspielplan/pokalwettbewerb/EM12/saison_id/2011',
            'https://www.transfermarkt.de/europameisterschaft-2016/gesamtspielplan/pokalwettbewerb/EM16/saison_id/2015',
            # AFC Asian Cup
            'https://www.transfermarkt.de/afc-asian-cup-2011/gesamtspielplan/pokalwettbewerb/AM11/saison_id/2010',
            'https://www.transfermarkt.de/afc-asian-cup-2015/gesamtspielplan/pokalwettbewerb/AM15/saison_id/2014',
            # Copa America
            'https://www.transfermarkt.de/copa-america-2011/gesamtspielplan/pokalwettbewerb/CA11/saison_id/2010',
            'https://www.transfermarkt.de/copa-america-2015/gesamtspielplan/pokalwettbewerb/CA15/saison_id/2014',
            # CONCACAF Cold Cup
            'https://www.transfermarkt.de/gold-cup-2011/gesamtspielplan/pokalwettbewerb/GC11/saison_id/2010',
            'https://www.transfermarkt.de/gold-cup-2015/gesamtspielplan/pokalwettbewerb/GC15/saison_id/2014',
            # Africa Cup of Nations
            'https://www.transfermarkt.de/afrika-cup-2013/gesamtspielplan/pokalwettbewerb/AC13/saison_id/2012',
            'https://www.transfermarkt.de/afrika-cup-2015/gesamtspielplan/pokalwettbewerb/AC15/saison_id/2014',
            'https://www.transfermarkt.de/afrika-cup-2017/gesamtspielplan/pokalwettbewerb/AC17/saison_id/2016'
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parseKOStage(self, row, game_type, tournament):
        item = None
        rowclass = row.xpath('@class').extract_first()
        if rowclass == "bg_Sturm":
            game_type = row.xpath('.//a/text()').extract_first()
        elif rowclass != 'bg_blau_20':
            date = row.xpath('td/text()').extract_first()
            date = re.sub(r"\s","",date)
            date = re.sub(r"[a-zA-Z]{2}\.*","",date)
            teams = row.xpath(
                './/a[contains(@class,"vereinprofil_tooltip")]/text()').extract()
            teamids = row.xpath(
                './/a[contains(@class,"vereinprofil_tooltip")]/@id').extract()
            result = row.xpath(
                './/a[contains(@class,"ergebnis-link")]/text()').extract_first().split(':')
            result_info = row.xpath(
                './/span[contains(@class,"ergebnis_zusatz")]/text()').extract_first()
            if result_info is not None:
                result_info = result_info.strip()
            else:
                result_info = ""
            game_id = row.xpath(
                './/a[contains(@class,"ergebnis-link")]/@id').extract_first()
            
            # convert abbreviation to full country name for match link
            teamA = search(teams[0])
            teamB = search(teams[1])
            teamidA = teamids[0]
            teamidB = teamids[2]
            item = items.GameListItem(
                tournament = tournament,
                gametype = game_type, 
                teamA = teamA,
                teamidA = teamidA,
                teamB = teamB,
                teamidB = teamidB,
                resultA = result[0],
                resultB = result[1],
                addinfo = result_info,
                gameid = game_id,
                date = date
            )
        return [item, game_type]
    def parseGroupStage(self, row, game_type, tournament):
            rowclass = row.xpath('@class').extract_first()
            if rowclass != 'bg_Sturm' and rowclass != 'bg_blau_20':
                date = row.xpath('td/text()').extract_first()
                date = re.sub(r"\s","",date)
                teams = row.xpath(
                    './/a[contains(@class,"vereinprofil_tooltip")]/text()').extract()
                teamids = row.xpath(
                    './/a[contains(@class,"vereinprofil_tooltip")]/@id').extract()
                result = row.xpath(
                    './/a[contains(@class,"ergebnis-link")]/text()').extract_first().split(':')
                result_info = row.xpath(
                    './/span[contains(@class,"ergebnis_zusatz")]/text()').extract_first()
                if result_info is not None:
                    result_info = result_info.strip()
                game_id = row.xpath(
                    './/a[contains(@class,"ergebnis-link")]/@id').extract_first()
                
                 # convert abbreviation to full country name for match link
                teamA = search(teams[0])
                teamB = search(teams[1])
                teamidA = teamids[0]
                teamidB = teamids[2]
                
                item = items.GameListItem(
                    tournament = tournament, 
                    gametype = game_type,
                    teamA = teamA,
                    teamidA = teamidA,
                    teamB = teamB,
                    teamidB = teamidB,
                    resultA = result[0],
                    resultB = result[1],
                    addinfo = result_info,
                    gameid = game_id,
                    date = date
                )                
                return item
    def parse(self, response):
        if response.status == 404:
            # stop crawling so error can be fixed
            print("ERROR for {url}".format(url=response.url))
            raise CloseSpider(reason='Could not access URL')
        tournament = response.url.split("/")[-3]
        game_type = ""
        for div in (
            response.css('.large-8.columns').xpath('div')):
                divclass = div.xpath('@class').extract_first()
                if( divclass == 'box'):
                    for row in div.xpath('table/tbody/tr'):
                        item, game_type = self.parseKOStage(row, game_type, tournament)
                        if item != None:
                            yield item
                elif( divclass == 'row'):
                    for group in div.css('.large-6.columns').xpath('div[contains(@class, "box")]'):
                        game_type = group.xpath('div[contains(@class, "table-header")]/text()').extract_first()
                        for row in group.xpath('table[2]/tbody/tr'):
                            yield self.parseGroupStage(row, game_type, tournament)