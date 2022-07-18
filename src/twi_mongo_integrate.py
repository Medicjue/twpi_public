def run():
    from mongo_dao import TwidataDAO
    from twitter_crawler import TwitterCrawler
    from datetime import datetime
    import logging
    
    crawler = TwitterCrawler(conf_path='conf/config.properties')

    today = datetime.now()
    today_str = today.strftime('%Y%m%d')
    logging.basicConfig(filename='log/twi_mongo_integrate_{}.log'.format(today_str), encoding='utf-8', level=logging.DEBUG)
    logging.info('Start execute crawler')

    keywords = ['國民黨', '民進黨', '民眾黨', '基進黨', '時代力量']
    dao = TwidataDAO()

    for keyword in keywords:
        logging.info('Keyword:{} start'.format(keyword))
        next_token = None

        sentences_set = set()
        raw_data = []
        cnt = 0
        while True:
            resp = crawler.run(keyword=keyword, next_token=next_token)
            next_token = resp['meta'].get('next_token', None)
            for entry in resp['data']:
                sentence = entry['text']
                if sentence not in sentences_set:
                    sentences_set.add(sentence)
                    entry['keyword'] = keyword
                    entry['parse_dt'] = today_str
                    raw_data.append(entry)
            if next_token is None:
                break
            elif cnt > 100:
                break
            else:
                cnt += 1

        
        ids = dao.insert_data(raw_data)
        logging.info('ids:{}'.format(ids))
        logging.info('Keyword:{} end'.format(keyword))


if __name__ == '__main__':
    run()

    