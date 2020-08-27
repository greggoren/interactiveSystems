from pymongo import MongoClient
from ShuffleUtils import retrieve_data
from scipy.stats import kendalltau
import numpy as np
def calculate_kt(init,scores_init,scores_modified):
    stats={}
    average = 0
    meadian = []
    for query in init:
        ranks_int = [scores_init[query][i] for i in init[query]]
        ranks_modified = [scores_modified[query][i] for i in init[query]]
        kt=kendalltau(ranks_int,ranks_modified)[0]
        stats[query]=kt
        average+=kt
        meadian.append(kt)
    average = average/len(stats)
    return average,stats,np.median(meadian)


def update_docs(stats):
    client = MongoClient('asr2.iem.technion.ac.il', 27017)
    db = client.asr16
    docs = db.interactive.find({})
    for doc in docs:
        consideration = "y"
        qid = doc["query_id"]
        if stats[qid]>0.7:
            consideration="n"
        doc["keep"]=consideration
        db.interactive.save(doc)


init_ranked_lists,_,__,scores_init = retrieve_data()
modified_ranked_lists,_,__,scores_modified = retrieve_data("modified")
average,stats,median = calculate_kt(init_ranked_lists,scores_init,scores_modified)
print("the average is: ",average)
print("the median is: ",median)
print("stats = ",stats)
# update_docs(stats)
