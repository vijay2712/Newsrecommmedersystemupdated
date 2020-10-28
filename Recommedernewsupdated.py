#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas

#Class for Popularity based Recommender System model
class popularity_recommender_py():
    
    def __init__(self):
        self.train_data = None
        self.popularity_recommendations = None
        
    #Create the popularity based recommender system model
    def create(self, train_data):
        self.train_data = train_data
        
        train_data=train_data.sort_values(['Rating'],ascending=[0])

    
        #Sort the news based upon recommendation score
        train_data['Ranking'] = train_data['Rating'].rank(ascending=0, method='first')
    
        #Generate a recommendation rank based upon score
        
        #Get the top 10 recommendations
        self.popularity_recommendations = train_data.head(6)
    def recommend(self, user_id):  
        
        user_recommendations = self.popularity_recommendations
        
        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user_id
    
        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]
        
        return user_recommendations
    
class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.item_similarity_recommendations = None
        
    #Get unique items (news) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Get unique users for a given item (news)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #Get unique items (news) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_news, all_news):
            
        ####################################
        #Get users for all news in user_news.
        ####################################
        user_news_users = []        
        for i in range(0, len(user_news)):
            user_news_users.append(self.get_item_users(user_news[i]))
            
        ###############################################
        #Initialize the item cooccurence matrix of size 
        #len(user_news) X len(news)
        ###############################################
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_news), len(all_news))), float)
           
        #############################################################
        #Calculate similarity between user news and all unique news
        #in the training data
        #############################################################
        for i in range(0,len(all_news)):
            #Calculate unique listeners (users) of news (item) i
            news_i_data = self.train_data[self.train_data[self.item_id] == all_news[i]]
            users_i = set(news_i_data[self.user_id].unique())
            
            for j in range(0,len(user_news)):       
                    
                #Get unique listeners (users) of news (item) j
                users_j = user_news_users[j]
                    
                #Calculate intersection of listeners of news i and j
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    #Calculate union of listeners of news i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix
    #Use the cooccurence matrix to make top recommendations
    
    def generate_top_recommendations(self, user, cooccurence_matrix, all_news, user_news):
        print("Non zero values in cooccurence_matrix :%d" % np.count_nonzero(cooccurence_matrix))
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value
        #Also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['user_id', 'Article_ID', 'score', 'rank']
        #index = np.arange(1) # array of numbers for the number of samples
        df = pandas.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_news[sort_index[i][1]] not in user_news and rank <= 10:
                df.loc[len(df)]=[user,all_news[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no news for training the item similarity based recommendation model.")
            return -1
        else:
            return df
    
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
        
    def recommend(self, user):
        
        ########################################
        #A. Get all unique news for this user
        ########################################
        user_news = self.get_user_items(user)    
            
        print("No. of unique news for the user: %d" % len(user_news))
        
        ######################################################
        #B. Get all unique items (news) in the training data
        ######################################################
        all_news = self.get_all_items_train_data()
        
        print("no. of unique news in the training set: %d" % len(all_news))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_news) X len(news)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_news, all_news)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_news, user_news)
                
        return df_recommendations
    
    def get_similar_items(self, item_list):
        
        user_news = item_list
        
        ######################################################
        #B. Get all unique items (news) in the training data
        ######################################################
        all_news = self.get_all_items_train_data()
        
        print("no. of unique news in the training set: %d" % len(user_news))
         
        ###############################################
        #C. Construct item cooccurence matrix of size 
        #len(user_news) X len(news)
        ###############################################
        cooccurence_matrix = self.construct_cooccurence_matrix(user_news, all_news)
        
        #######################################################
        #D. Use the cooccurence matrix to make recommendations
        #######################################################
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_news, user_news)
         
        return df_recommendations
 


# In[ ]:




