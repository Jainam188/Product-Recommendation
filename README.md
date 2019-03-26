# Product-Recommendation
Recommendation System

Short Introduction

    Popularity Model

 Recommend items which are liked or bought by most number of users. but there is no personalization recommendation involved with this approach.
 popularity is defined on the entire user pool.
 
 for example, if some app or website recommend you a product just because that product is liked by other users and it doesn't care that you are interested in buying that product.
 
    Collaborative Model
    
 The main idea of collaborative model is that it finds similarities between products of different users and recommend products.
 
 for example, if X user has bought A,B,C products and another Y user has bought B,C,D products, so X should like D product and Y user should like A. Because that have similarity of B,C products.
 
I have applied both Model for product recommendation also uploading the dataset with it.

I used two dataset

    1. data.csv for train my both model
    2. custid.csv for recommending products to selected Customer Ids.

In Product Recommendation it will recommend 10 product to every user.

    Dependancies
Import Pandas

Import Graphlab

For more understanding of recommendation refer below link

https://www.analyticsvidhya.com/blog/2016/06/quick-guide-build-recommendation-engine-python/
