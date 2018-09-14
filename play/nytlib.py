import requests
import json
import newspaper

def get_json(n, query, desk, begin_date, end_date ):
    '''
    This function accepts n(number of pages) ,query string, desk, begin and end dates as parameters.
    The date format is 20180328
    The query is the search string and desk is the news desk to use.
    It returns a json file for the all the articles returned by the New York Time.
    '''
    url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    all_articles = []
    for i in range(0,n): #NYT limits pager to first 100 pages. But rarely will you find over 100 pages of results anyway.
        params = {'q':query,
                  'fq':{'source': ['The New York Times'], 'news_desk' : query},
    'begin_date' : begin_date,
    'end_date' : end_date,
    'api-key' : '15f8c1ae34044c32974dc9918dfd350a',
    'sort':'oldest',
    'page':str(i)}
        resp_art = requests.get(url = url,params = params)
        articles_art = resp_art.json()
        articles = parse_articles(articles_art)
        all_articles = all_articles + articles
    return(all_articles)

def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    for i in articles['response']['docs']:
        dic = {}
        dic['id'] = i['_id']
        #if i['abstract'] is not None:
            #dic['abstract'] = i['abstract'].encode("utf8")
        dic['headline'] = i['headline']['main'].encode("utf8")
        #dic['desk'] = i['news_desk']
        dic['date'] = i['pub_date'][0:10] # cutting time of day.
        #dic['section'] = i['section_name']
        if i['snippet'] is not None:
            dic['snippet'] = i['snippet'].encode("utf8")
        #dic['source'] = i['source']
        #dic['type'] = i['type_of_material']
        dic['url'] = i['web_url']
        dic['word_count'] = i['word_count']
        # locations
        #locations = []
        #for x in range(0,len(i['keywords'])):
            #if 'glocations' in i['keywords'][x]['name']:
                #locations.append(i['keywords'][x]['value'])
        #dic['locations'] = locations
        # subject
        #subjects = []
        #for x in range(0,len(i['keywords'])):
            #if 'subject' in i['keywords'][x]['name']:
                #subjects.append(i['keywords'][x]['value'])
        #dic['subjects'] = subjects   
        news.append(dic)
    return(news) 

def get_articles(art, fldr):
    '''
    The following function saves the articles in the given directory
    saving title and article in seprate files.
    It uses Newspaper library.
    Takes two parameter article and saving directory.
    '''
    i = 0
    ec = 0
    for t in range(len(art)) :
        try :
            article = newspaper.Article(art[t]['url'], language = 'en')
            article.download()
            article.parse()
            if(len(article.text) > 100) :
                title = 'title-'+str(i)+'.txt'
                content = 'article-'+str(i)+'.txt'
                with open(fldr+title,'w+') as file_1:
                    file_1.write(article.title)
                with open(fldr+content,'w+') as file_2:
                    file_2.write(article.text)
                i += 1
        except :
            print('E_C : ',ec)
            ec += 1
            pass
