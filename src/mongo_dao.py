from configparser import ConfigParser
from http import client

import pymongo

class MongoDBConf:
    def __init__(self):
        config = ConfigParser()
        config.read('conf/config.properties')
        acct = config['mongo']['acct']
        end_point = config['mongo']['endpoint']
        pwd = config['mongo']['pwd']
        self._myclient = pymongo.MongoClient("mongodb+srv://{}:{}@{}/test".format(acct, pwd, end_point))

class TwidataDAO(MongoDBConf):
    def __init__(self):
        super().__init__()
        self.__mydb = self._myclient["raw_data"]
        self.__mycol = self.__mydb['twidata']

    def insert_data(self, data:list) -> list:
        ids = self.__mycol.insert_many(data)
        return ids

    def query(self, keyword_in_text:str=None, parse_dt:str=None) -> list:
        query = {}
        if keyword_in_text is not None:
            query_string = {'$regex':'.*'+keyword_in_text+'.*'}
            query['text'] = query_string
        if parse_dt is not None:
            query['parse_dt'] = parse_dt
        return self.__mycol.find(query)

class IndexDAO(MongoDBConf):
    def __init__(self):
        super().__init__()
        self._mydb = self._myclient["index"]

class SentimentIndexDAO(IndexDAO):
    def __init__(self):
        super().__init__()
        self.__mycol = self._mydb['sentiment']

    def insert_data(self, data:list) -> list:
        ids = self.__mycol.insert_many(data)
        return ids

    def query(self, keyword_in_text:str=None, parse_dt:str=None) -> list:
        query = {}
        if keyword_in_text is not None:
            query_string = {'$regex':'.*'+keyword_in_text+'.*'}
            query['text'] = query_string
        if parse_dt is not None:
            query['parse_dt'] = parse_dt
        return self.__mycol.find(query)

if __name__ == '__main__':
    dao = TwidataDAO()
    data = dao.query('韓總')
    for x in data:
        print(x)
