

class Crawler:
    def run(self, keyword:str) -> list:
        pass


class TwitterCrawler(Crawler):
    __endpoint_template = 'https://api.twitter.com/2/tweets/search/recent?query={}'
    def __init__(self, conf_path:str) -> None:
        from configparser import ConfigParser
        super().__init__()
        config = ConfigParser()
        config.read(conf_path)
        self.__bearer_token = config['twitter']['bearer_token']

    def run(self, keyword:str, next_token:str=None) -> list:
        import requests
        import urllib.parse
        import json
        endpoint = self.__endpoint_template.format(urllib.parse.quote(keyword))
        if next_token is not None:
            endpoint += '&next_token=' + next_token
        headers = {"Authorization": "Bearer "+self.__bearer_token}
        resp = requests.get(endpoint, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        else:
            return []

if __name__ == '__main__':
    crawler = TwitterCrawler(conf_path='conf/config.properties')
    r = crawler.run('國民黨')
    print(r)