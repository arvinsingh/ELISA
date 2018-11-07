import spacy
from pyspark.sql import SparkSession

def setup():
    # load spacy model for feature generation
     nlp = spacy.load('en_core_web_md')
     print("-----> Spacy model loaded!!!")

     base_url = 'hdfs:///ELISA/'
     sparkSession = SparkSession.builder.appName("ELISA_pyspark").getOrCreate()
     
     return nlp