# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class LineupItem(scrapy.Item):
    gameid = scrapy.Field()
    team = scrapy.Field()
    club = scrapy.Field()
    name = scrapy.Field()
    age = scrapy.Field()
    position = scrapy.Field()
    value = scrapy.Field()
    start = scrapy.Field()

class GameListItem(scrapy.Item):
    tournament = scrapy.Field()
    gametype = scrapy.Field()
    teamA = scrapy.Field()
    teamidA = scrapy.Field()
    teamB = scrapy.Field()
    teamidB = scrapy.Field()
    resultA = scrapy.Field()
    resultB = scrapy.Field()
    addinfo = scrapy.Field()
    gameid = scrapy.Field()
    date = scrapy.Field()

class CompareItem(scrapy.Item):
    for_game = scrapy.Field()
    gametype = scrapy.Field()
    gamecontext = scrapy.Field()
    gamedate = scrapy.Field()
    teamidA = scrapy.Field()
    teamidB = scrapy.Field()
    resultA = scrapy.Field()
    resultB = scrapy.Field()