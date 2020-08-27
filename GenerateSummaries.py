from pymongo import MongoClient
import sys

def read_features(fname):
    features = {}
    with open(fname) as f:
        for row in f:

            name = row.split(" # ")[1].rstrip()
            feature_values = [float(i.split(':')[1].rstrip()) for i in row.split(" # ")[0].split()[2:]]
            features[name]=feature_values
    return features

def read_model(fname):
    model = []
    with open(fname) as f:

        values = f.readlines()[-1].replace("#","")
        last_index = 0
        for pair in values.split()[1:]:
            index = int(pair.split(":")[0])
            value = float(pair.split(":")[1].rstrip())
            if index>last_index+1:
                for i in range(last_index+1,index):
                    model.append(0)
            model.append(value)
            last_index=index
    return model

featureNames = [
    'docCoverQueryNum',
    'docCoverQueryRatio',
    'docLen',
    'docIDF',
    'docBM25',
    'docLMIR.DIR',
    'docLMIR.JM',
    'docEnt',
    'docStopCover',
    'docFracStops',
    'docTF',
    'docTFNorm',
    'docTFIDF', ]

def calculate_weighted_sum(model,featues_line):
    stats = {"quality":0,"term_freq":0,"lm":0}
    quality_indices = [2,7,8,9]
    term_freq_indices = [0,1,4,]
    term_freq_indices.extend([i for i in range(9,len(model))])
    language_model_indices = [5,6]

    for index in quality_indices:
        stats["quality"]+=model[index]*featues_line[index]
    for index in term_freq_indices:
        stats["term_freq"]+=model[index]*featues_line[index]
    for index in language_model_indices:
        stats["lm"]+=model[index]*featues_line[index]

    return stats

def get_relative_scores(model,features):
    stats = {}
    for doc in features:
        features_line = features[doc]
        query = doc.split("-")[2]
        if query not in stats:
            stats[query]={}
        stats[query][doc]=calculate_weighted_sum(model,features_line)
    return stats

def calcualte_relative_performance(weighted_scores):
    ranked_docs = {}
    for query in weighted_scores:
        if query not in ranked_docs:
            ranked_docs[query]={}

        ranked_docs[query]["quality"] = sorted(list(weighted_scores[query].keys()),key=lambda x:weighted_scores[query][x]["quality"])
        ranked_docs[query]["lm"] = sorted(list(weighted_scores[query].keys()),key=lambda x:weighted_scores[query][x]["lm"])
        ranked_docs[query]["term_freq"] = sorted(list(weighted_scores[query].keys()),key=lambda x:weighted_scores[query][x]["term_freq"])
    return ranked_docs


def generate_summary(ranked_docs,docname,query):
    stats={}
    keys = list(ranked_docs[query].keys())
    for key in keys:
        value = round(ranked_docs[query][key].index(docname)/(len(ranked_docs[query][key])),2)*100
        stats[key]=value
    summary = "This document has: \n 1) higher score in terms of quality features than:"+str(stats["quality"])+"% of documents in the list \n"+" 2) higher score in terms of term frequency features than:"+str(stats["term_freq"])+"% of documents in the list  \n"+" 3) higher score in terms of language modeling features than:"+ str(stats["lm"])+"% of documents in the list"
    return summary
    




def generate_summaries(ranked_docs,status="init"):
    client = MongoClient('asr2.iem.technion.ac.il', 27017)
    db = client.asr16
    docs = db.interactive.find({"status":status})
    for doc in docs:
        query = doc["query_id"]
        docname = doc["docname"]
        summary = generate_summary(ranked_docs,docname,query)
        doc["summary"] = summary
        db.interactive.save(doc)


if __name__=="__main__":
    base_dir = "base/"
    features_fname = sys.argv[1]
    status = sys.argv[2]
    model_fname = base_dir+"Code/Models/experiment_model"
    features = read_features(features_fname)
    model = read_model(model_fname)
    weighted_scores = get_relative_scores(model,features)
    relative_performance = calcualte_relative_performance(weighted_scores)
    generate_summaries(relative_performance,status)







