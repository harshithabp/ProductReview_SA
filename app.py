from flask import Flask,request, url_for, redirect, render_template,Markup
import pickle
import numpy as np
from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import requests as rq
import nltk
import emoji
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import os
from selenium import webdriver

op=webdriver.ChromeOptions()
op.binary_location=os.environ.get("GOOGLE_CHROME_BIN")
op.add_argument("--headless")
op.add_argument("--no-sandbox")
op.add_argument("--disable-dev-sh-usage")

driver=webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"),chrome_options=op)

import re




application = app = Flask(__name__) 

model=pickle.load(open('model.pkl','rb'))
vect=pickle.load(open('vect.pkl','rb'))

review_star_rating=[]



  
@app.route('/') 
def index():
   #return "WELCOME!!! This is the home page" 
    lable=["January","February","March","April","May","June"]
    data=[203,156,99,251,305,247]
    return render_template("review.html")

@app.route('/home') 
def index1():
   return "WELCOME!!!"

@app.route('/predict',methods=['POST','GET'])
def predict():
    reviews=[]
    reviewtitle=[]
    review_star_rating=[]
    if request.method == 'POST':
        result = request.form['Review_url']
        #driver = webdriver.Chrome(r'C:\Users\harshitha\Downloads\chromedriver_win321\chromedriver.exe')
        page=result + "&pageNumber"
        for x in range(1,250):
            base_url=page+"={}".format(x)
            driver.get(base_url)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
        #soup = BeautifulSoup(data, 'html.parser')
            reviewcontainer=soup.findAll('div',{'class':'a-section review aok-relative'})
            for container in reviewcontainer:
                review=container.find('span',{'data-hook':'review-body'}).span
                if review:
                    reviews.append(review.text)
                else:
                    reviews.append(None)
        #review title
                review_title=container.find('a',{'data-hook':'review-title'}).span
                if review_title:
                    reviewtitle.append(review_title.text)
                else:
                    reviewtitle.append(None)
        #ratings
                rating=container.find('i',{'data-hook':'review-star-rating'}).span
                if rating:
                    review_star_rating.append(rating.text)
                else:
                    review_star_rating.append(None)
            #print(base_url)
            try:
                driver.find_element_by_partial_link_text('Next page').click()
            except:
                break
        rr=[]
        d = {'reviews':reviews,'review_title':reviewtitle,'rating':review_star_rating} 

        df = pd.DataFrame(d)
        
         
        test_features = vect.transform([r for r in df.reviews])
        y_pred = model.predict(test_features)
        print(y_pred)
        
        pos_count=0
        neg_count=0
        #polarity score
        rr=[]
        
        for count,y in enumerate(y_pred):
            if y=='Negative':
                neg_count += 1
                

            elif y=='Positive':
                pos_count += 1
                 

        sa = SentimentIntensityAnalyzer()        
        for item in df.reviews:
            ps=sa.polarity_scores(item)['compound']
            rr.append(ps)

        #dataframe
        d = {'reviews':reviews,'review_title':reviewtitle,'rating':review_star_rating,'polarity':rr,'sentiment':y_pred} 
        df = pd.DataFrame(d)

        #extracting numerical values from rating (5 star out of 5 ---> 5)
        df["rating"]=pd.DataFrame(df.rating.str.split(' ',1).tolist())

        #df of positive reviews
        df1=df.loc[df['sentiment'] == 'Positive']
        print(df1)

        #df of negative reviews
        df2=df.loc[df['sentiment'] == 'Negative']
        print(df2)

        #sorting pos reviews dataframe in descending order
        pol_pos = df1['polarity'].gt(0)
        df_pos = pd.concat([df1[pol_pos].sort_values('polarity',ascending=False),df1[~pol_pos].sort_values('polarity', ascending=False)], ignore_index=True)
        #top 10 pos reviews
        top_pos=df_pos[:10]
        print(top_pos['polarity'])

        #sorting neg reviews dataframe in descending order
        pol_neg = df2['polarity'].gt(0)
        df_neg = pd.concat([df2[pol_neg].sort_values('polarity',ascending=False),df2[~pol_neg].sort_values('polarity', ascending=False)], ignore_index=True)#negative sentiment df
        #top 10 pos reviews
        top_neg=df_neg[:-11:-1]
        print(top_neg['polarity'])



        labels=['positive','negative']
        values=[pos_count,neg_count] 
        colors=['#dee7ce','#ff471a']
        return render_template('review.html' ,lables1=top_neg["rating"],data1=top_neg["polarity"],lables=top_pos["rating"],data=top_pos["polarity"],max=50000,result = '{}'.format(pos_count+neg_count),pos='{}%'.format(int((pos_count/(pos_count+neg_count))*100)),neg='{}%'.format(int((neg_count/(pos_count+neg_count))*100)),set=zip(values, colors),posrev=zip(top_pos.review_title,top_pos.rating),negrev=zip(top_neg.review_title,top_neg.rating),neg_revt=zip(top_neg.reviews,top_neg.review_title,top_neg.rating,top_neg.polarity),pos_revt=zip(top_pos.reviews,top_pos.review_title,top_pos.rating,top_pos.polarity))
    else:
        return render_template('review.html')

  
if __name__ == '__main__': 
    app.debug = True
    app.run()