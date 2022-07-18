from datetime import datetime, timedelta
import pandas as pd

def run(target_dt:str=None):
    from mongo_dao import TwidataDAO, SentimentIndexDAO
    from sentiment_judger import Judger
    today = datetime.now()
    today_str = today.strftime('%Y%m%d')
    if target_dt is None:
        yesterday = today - timedelta(days=1)
        target_dt = yesterday.strftime('%Y%m%d')

    keywords = ['國民黨', '民進黨', '民眾黨', '基進黨', '時代力量']

    dao = TwidataDAO()
    judger = Judger(model_path='bookingReview/bookingReviewMdl/', words_index_path='bookingReview/words_index.pickle')

    daily_data = []
    sentences = []

    for keyword in keywords:
        print('Start collecting {}'.format(keyword))
        result = dao.query(keyword_in_text=keyword, parse_dt=target_dt)
        for e in result:
            entry = {'agg_dt':today_str,'parse_dt':e['parse_dt'], 'keyword':keyword, 'id':e['id']}
            raw_text = e['text']
            sentences.append(raw_text)
            entry['text'] = raw_text
            daily_data.append(entry)
    print('Start inference')
    scores = judger.judge(sentences=sentences)
    daily_df = pd.DataFrame(daily_data)
    score_series = pd.Series(scores[:,-1])
    daily_df['score'] = score_series
    daily_df.to_csv('output/{}_raw.csv'.format(target_dt))
    daily_agg_df = daily_df.groupby(['keyword']).agg({'score': 'count', 'score': 'mean', 'score': lambda x: x.sum()*x.count()}).reset_index()
    daily_agg_df['dt'] = target_dt
    daily_agg_df['mdl_ver'] = '0.0.1'
    daily_agg_df.to_csv('output/{}_agg.csv'.format(target_dt))

    to_be_uploaded = []
    for _, row in daily_agg_df.iterrows():
        to_be_uploaded.append(row.to_dict())
    index_dao = SentimentIndexDAO()
    index_dao.insert_data(to_be_uploaded)

if __name__ == '__main__':
    import sys
    print(sys.argv)
    if len(sys.argv) > 1:
        run(target_dt=sys.argv[1])
    else:
        run()
