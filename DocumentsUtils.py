import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import sample
from pymongo import MongoClient
import numpy as np
from RankingUtils import create_trectext_for_current_documents, merge_indices, build_index,run_ranking_model


def get_vectors(strs):
    strs = [clean_texts(text) for text in strs]
    vectorizer = CountVectorizer(strs)
    vectors = vectorizer.fit_transform(strs)
    return vectors.toarray()


def retrieve_initial_documents(fname):
    initial_query_docs = {}
    tree = ET.parse(fname)
    root = tree.getroot()
    queries = set()
    for doc in root:
        name = ""
        for att in doc:
            if att.tag == "DOCNO":
                name = att.text
                query = name.split("-")[2]
                queries.add(query)
                r = name.split("-")[1]
                if r not in initial_query_docs:
                    initial_query_docs[r] = {}
                if query not in initial_query_docs[r]:
                    initial_query_docs[r][query] = {}
            else:
                initial_query_docs[r][query][name] = att.text
    return initial_query_docs, queries


def clean_texts(text):
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = text.replace(",", " ")
    text = text.replace(":", " ")
    text = text.replace("?", " ")
    text = text.replace("]", "")
    text = text.replace("[", "")
    text = text.replace("}", "")
    text = text.replace("{", "")
    text = text.replace("+", " ")
    text = text.replace("~", " ")
    text = text.replace("^", " ")
    text = text.replace("#", " ")
    text = text.replace("$", " ")
    text = text.replace("!", "")
    text = text.replace("|", " ")
    text = text.replace("%", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("\\", " ")
    text = text.replace("*", " ")
    text = text.replace("&", " ")
    text = text.replace(";", " ")
    text = text.replace("`", "")
    text = text.replace("'", "")
    text = text.replace("@", " ")
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    text = text.replace("/", " ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    return text.lower()


def check_similarity_to_population(candidate_text, all_texts):
    all_texts.append(candidate_text)
    all_vectors = get_vectors(all_texts)
    canidate_vector = all_vectors[-1]
    all_vectors = all_vectors[:-1]
    similarities = []
    for vec in all_vectors:
        similarities.append(cosine_similarity([vec], [canidate_vector])[0][0])

    return max(similarities)


def choose_docs_for_experiment(texts, queries, rounds):
    experiment_texts = {}
    for r in rounds:
        for query in queries:
            if query not in experiment_texts:
                experiment_texts[query] = {}
                chosen_doc = sample(list(texts[r][query].keys()), 1)[0]
            else:
                current_scores = {}
                current_docs = [experiment_texts[query][doc] for doc in experiment_texts[query]]
                for doc in texts[r][query]:
                    text = texts[r][query][doc]
                    similarity = check_similarity_to_population(text, current_docs)
                    current_scores[doc] = similarity
                chosen_doc = sorted(list(current_scores.keys()), key=lambda x: current_scores[x])[0]
            experiment_texts[query][chosen_doc] = texts[r][query][chosen_doc]
    return experiment_texts


def upload_data_mongo(texts, scores, queries,status = "init"):
    client = MongoClient('asr2.iem.technion.ac.il', 27017)
    db = client.asr16
    for query in texts:
        for doc in texts[query]:
            # query = doc.split("-")[2]
            obj = {}
            obj["text"] = texts[query][doc]
            obj["docname"] = doc
            obj["query_id"] = query
            obj["query"] = queries[query]
            obj["score"] = scores[doc]
            obj["status"]=status
            db.interactive.insert(obj)


def read_scores(scores_file):
    scores = {}
    with open(scores_file) as f:
        for line in f:
            docname = line.split()[2]
            score = float(line.split()[4])
            scores[docname]=score
    return scores


def read_queries_text(fname):
    queries = {}
    with open(fname) as f:
        for line in f:
            qid = line.split(":")[0]
            query_text = line.split(":")[1].rstrip()
            queries[qid] = query_text
    return queries



if __name__ == "__main__":
    base_dir = "base/"
    query_text_file = base_dir+"data/queris.txt"
    texts, queries = retrieve_initial_documents("documents.trectext")
    rounds = ["01", "03", "05", "06", "08"]
    experiment_docs = choose_docs_for_experiment(texts, queries, rounds)
    trectext_filename, workingset_filename, current_time = create_trectext_for_current_documents(base_dir, experiment_docs)
    new_index = build_index(trectext_filename, current_time, base_dir)
    merged_index = merge_indices(asr_index=new_index, base_dir=base_dir)
    ranked_list = run_ranking_model(merged_index, workingset_filename, current_time, base_dir)
    scores = read_scores(ranked_list)
    queries_text = read_queries_text(query_text_file)
    upload_data_mongo(experiment_docs,scores,queries_text)