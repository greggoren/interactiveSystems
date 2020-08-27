from krovetzstemmer import Stemmer
from DocumentsUtils import clean_texts,read_scores,upload_data_mongo
from random import random
from pymongo import MongoClient
from RankingUtils import create_trectext_for_current_documents,build_index,run_ranking_model_diff

def modify_text(text,index,query):
    stemmer =Stemmer()
    query_terms = [stemmer.stem(q) for q in query.split()]
    new_text = ""

    if index==4:
        new_text=query+text+query
        return new_text

    elif index==0:
        p = 0.5
    elif index==2:
        p = 0.2

    tokens = clean_texts(text).split()

    for token in tokens:
        if stemmer.stem(token) in query_terms:
            if random() < p:
                continue
        new_text += token + " "
    return new_text

def modify_documents(texts,queries,ranked_lists):
    for query in ranked_lists:
        current_list = ranked_lists[query]
        modification_indexes = [0,2,4]
        for index in modification_indexes:
            doc = current_list[index]
            text = texts[query][doc]
            query_text = queries[query]
            new_text = modify_text(text,index,query_text)
            texts[query][doc]=new_text
    return texts


def retrieve_data(status="init"):
    client = MongoClient('asr2.iem.technion.ac.il', 27017)
    db = client.asr16
    docs = db.interactive.find({"status":status})
    stats={}
    texts = {}
    queries_text = {}
    for doc in docs:

        docname = doc["docname"]
        qid = doc["query_id"]
        if qid not in stats:
            stats[qid]={}
            texts[qid]={}
        score = doc["score"]
        texts[qid][docname]=doc["text"]
        queries_text[qid]=doc["query"]

        stats[qid][docname]=score
    ranked_lists = {}
    for qid in stats:
        ranked_lists[qid]=sorted(list(stats[qid].keys()),key=lambda x:stats[qid][x],reverse=True)
    return ranked_lists,texts,queries_text,stats





if __name__=="__main__":
    base_dir = "base/"
    ranked_lists,texts,queries_text,_ = retrieve_data()
    modified_docs = modify_documents(texts,queries_text,ranked_lists)
    trectext_filename, workingset_filename, current_time = create_trectext_for_current_documents(base_dir,modified_docs)
    index = build_index(trectext_filename,current_time,base_dir)
    merged_index = base_dir+'Collections/mergedindex'
    modified_ranked_lists_file = run_ranking_model_diff(merged_index,workingset_filename,current_time,base_dir,index)
    scores = read_scores(modified_ranked_lists_file)
    upload_data_mongo(modified_docs,scores,queries_text,"modified")









