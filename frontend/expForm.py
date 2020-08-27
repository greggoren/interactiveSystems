from django import forms
from pymongo import MongoClient

def getQueryText():
	queries = []
	client = MongoClient()
	db = client.asr16
	docs = db.interactive.find({"status":"init","keep":"y"})
	seen = []
	for doc in docs:
		qid = doc["query_id"]
		if qid not in seen:
			queries.append((qid+"-"+doc["query"],doc["query"]))
			seen.append(qid)
	return tuple(queries)



class QueryForm(forms.Form):
	QUERIES = getQueryText()
	query = forms.ChoiceField(choices=QUERIES,label="Your query")
	
class FinalForm(forms.Form):
	name = forms.CharField(label='Your name',widget=forms.TextInput(attrs={'size': '10'}))
	age = forms.IntegerField(label='Your age',min_value=1)
	rate_exp = forms.IntegerField(label='Rate the overall experience of using the search engine',min_value=1,max_value=5)
	rate_trust = forms.IntegerField(label='Rate your trust level with respect to the search engine',min_value=1,max_value=5)
	rate_stable = forms.IntegerField(label='Rate your opinion of the search engine\'s stability in terms of changes in rankings',min_value=1,max_value=5)

