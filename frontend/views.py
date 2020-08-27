from django.shortcuts import render, render_to_response
from django.template import RequestContext
from django.http import HttpResponse
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
import datetime
from numpy import mean
import json
from bson.son import SON
from expForm import QueryForm, FinalForm
from random import random
# Create your views here.

def home(request):
	countdown = getCountdownToMidnight()
	indeterminate = getIndeterminate()
	return render(request, "index.html", {'countdown':countdown, 'indeterminate':indeterminate,})

def rules(request):
	countdown = getCountdownToMidnight()
	indeterminate = getIndeterminate()
	return render(request, "rules.html", {'countdown':countdown, 'indeterminate':indeterminate,})

def awards(request):
	countdown = getCountdownToMidnight()
	indeterminate = getIndeterminate()
	username = getUsername(request)
	winning = getNumberOfWinningGames(username)
	return render(request, "awards.html", {'countdown':countdown, 'indeterminate':indeterminate, 'winning':winning })
	
def staff(request):
	countdown = getCountdownToMidnight()
	client = MongoClient()
	db = client.asr16
	documents = db.documents.find({})
	updatedDocuments, avgAdditions, avgDeletions, avgTotal = getNumberOfUpdatedDocuments(documents)
	winnerDocuments = db.documents.find({'position':1})
	updatedDocumentsWinner, avgAdditionsWinner, avgDeletionsWinner, avgTotalWinner = getNumberOfUpdatedDocuments(winnerDocuments)
	nonWinnerDocuments = db.documents.find({'position':{ '$ne':1 }})
	updatedDocumentsNonWinner, avgAdditionsNonWinner, avgDeletionsNonWinner, avgTotalNonWinner = getNumberOfUpdatedDocuments(nonWinnerDocuments)
	indeterminate = getIndeterminate()
	return render(request, "staff.html", {'countdown':countdown, 'indeterminate':indeterminate,
		'updated_documents':updatedDocuments, 'avg_additions':avgAdditions, 'avg_deletions':avgDeletions, 'avg_total':avgTotal,
		'updated_documents_winner':updatedDocumentsWinner, 'avg_additions_winner':avgAdditionsWinner, 'avg_deletions_winner':avgDeletionsWinner, 'avg_total_winner':avgTotalWinner,
		'updated_documents_non_winner':updatedDocumentsNonWinner, 'avg_additions_non_winner':avgAdditionsNonWinner, 'avg_deletions_non_winner':avgDeletionsNonWinner, 'avg_total_non_winner':avgTotalNonWinner})
	
def documents(request):
	countdown = getCountdownToMidnight()
	username = getUsername(request)
	error_message = countNumberOfTokens(request, username)
	documents = getUserDocuments(username)
	indeterminate = getIndeterminate()
	return render(request, "documents.html", {'error_message':error_message, 'username':username, 'documents': documents, 'countdown':countdown, 'indeterminate':indeterminate})

def queries(request, query):
	countdown = getCountdownToMidnight()
	username = getUsername(request)
	documents = getQueryDocuments(query,username)
	indeterminate = getIndeterminate()
	return render(request, "queries.html", { 'documents':documents, 'username':username, 'query':query, 'countdown':countdown, 'indeterminate':indeterminate})

def featureView(request, featureName):
	countdown = getCountdownToMidnight()
	return render(request, "featureView.html", { 'iteration_labels':json.dumps(getIterationLabels()), 'featureTitle':featureName, 'feature':getFeaturesForEachType(featureName), 'countdown':countdown})
	
def getUsername(request):
	"""
	Retreive the username from the request.
	"""
	username = None
	if request.user.is_authenticated():
		username = request.user.username
	return username

def countNumberOfTokens(request, username):
	"""
	Count the number of tokens for a document and return an error msg if needed.
	"""
	error_message = 0
	if request.method == 'POST':
		query = request.POST.get('query', '')
		document = request.POST.get(query, '')
		try:
			vectorizer = CountVectorizer(min_df=1)
			X = vectorizer.fit_transform([str(document)])
			numberOfTokens = sum(X.toarray()[0])
		except ValueError:
			numberOfTokens = 0
		if int(numberOfTokens)==0 or int(numberOfTokens)>150:
			error_message = "The number of tokens is illegal: " + str(numberOfTokens)
		else:
			updateDocument(username, query, document)
	return error_message

def getUserDocuments(username):
	if username is None:
		return {}
	client = MongoClient()
	db = client.asr16
	documents = db.documents.find({'username':username})
	return documents

def getQueryDocuments(query,username):
	client = MongoClient()
	db = client.asr16
	#documents = db.documents.find({'query':query}).sort("position", 1)
	query_id = db.documents.find({'query':query,'username':username})[0]['query_id']
	documents=db.documents.find({'query_id':query_id}).sort("position", 1)
	return documents

def updateDocument(username, query, document):
	client = MongoClient()
	db = client.asr16
	indeterminate = getIndeterminate()
	if not indeterminate:
		documentToUpdate = db.documents.find_one({'username':username, 'query':query})
		documentToUpdate['current_document'] = document
		current_time = datetime.datetime.now()
		documentToUpdate['edittion_time'] = current_time
		db.documents.save(documentToUpdate)
	return
	
def getCountdownToMidnight():
	NO_MONDAY_OR_THURSDAY = 0
	now = datetime.datetime.now()
	#if now.weekday()!=2:
	return NO_MONDAY_OR_THURSDAY
	mnight = now.replace(hour=22, minute=0, second=0, microsecond=0)
	return int((mnight-now).seconds)

	
def getNumberOfTokensAddAndRemoved(text, text_before):
	vectorizer = CountVectorizer(min_df=1)
	X = vectorizer.fit_transform([str(text_before),str(text)])
	resultedVector = X[1] - X[0]
	deletions = 0
	additions = 0
	for x in resultedVector.T.toarray():
		if x[0]>0 :
			additions += x[0]
		elif x[0]<0 :
			deletions += -1*x[0]
	numberOfTokens = additions + deletions
	return numberOfTokens, additions, deletions

def getNumberOfUpdatedDocuments(documents):
	documentsChanged = 0
	totalDocuments = 0
	totalAdditions = []
	totalDeletions = []
	total = []
	for document in documents:
		numberOfTokens, additions, deletions = getNumberOfTokensAddAndRemoved(document['current_document'],document['posted_document'])
		if numberOfTokens > 0:
			documentsChanged += 1
		totalAdditions.append(additions)
		totalDeletions.append(deletions)
		total.append(numberOfTokens)
		totalDocuments +=1
	return documentsChanged, sum(totalAdditions)/float(len(totalAdditions)), mean(totalDeletions), mean(total)
	# return documentsChanged, sum(totalAdditions)/float(totalDocuments), sum(totalDeletions)/float(totalDocuments), sum(total)/float(totalDocuments)

def getIterationLabels():
	client = MongoClient()
	db = client.asr16
	iterations = db.archive.distinct("iteration")
	iterationLabels = []
	for iteraion in iterations:
		year = iteraion.split("-")[0]
		month = iteraion.split("-")[1]
		day = iteraion.split("-")[2]
		iterationLabels.append(day + "/" + month)
	return iterationLabels

	
def aggregateFeatures(featureName, match = []):
	client = MongoClient()
	db = client.asr16
	aggregationSettings = match + [ 
		{ "$unwind" : "$features." + featureName },
		{ "$group": { "_id": {"iteration":"$iteration", "query":"$query_id"}, "total": { "$avg": "$features." + featureName } } },
		{ "$group": { "_id": "$_id.iteration", "total": { "$avg": "$total" } } },
		{ "$sort": SON([("_id",1)])}
	]
	results = list(db.archive.aggregate(aggregationSettings))
	values = []
	for val in results:
		values.append(val["total"])
	return values

def getFeaturesForEachType(featureName):
	result = { 
		'All': aggregateFeatures(featureName), 
		'Winners':aggregateFeatures(featureName, match = [{"$match": { "position":1 }}]),
		'NonWinners': aggregateFeatures(featureName, match = [{"$match": { "position":{ "$ne": 1 } }}])
		}
	return result

def getIndeterminate():
	client = MongoClient()
	db = client.asr16
	status = db.status.find_one({})
	return status['indeterminate']
	

def calculateBonus(username):
	client = MongoClient()
	db = client.asr16
	bonus = db.archive.count({'username':username, 'position':1})+0.5*db.archive.count({'username':username, 'position':2})+(0.33)*db.archive.count({'username':username,'position':3})
	return bonus

def getNumberOfWinningGames(username):
	client = MongoClient()
	db = client.asr16
	#winning = db.archive.count({'username':username, 'position':1})+0.5*db.archive.count({'username':username, #'position':2})+(0.33)*db.archive.count({'username':username, 'position':3})
	winning = calculateBonus(username)
	return winning
	
def chooseQuery(request):
    # if this is a POST request we need to process the form data
	if request.method == 'POST':
	# create a form instance and populate it with data from the request:
		form = QueryForm(request.POST)
        # check whether it's valid:
		if form.is_valid():
			query_data = form.cleaned_data['query']
			query = query_data.split("-")[0]
			query_text = query_data.split("-")[1]
			client = MongoClient()
			db = client.asr16
			documents = db.interactive.find({"query_id":query,"status":"init"}).sort([("score",-1),("docname",1)])
			p = random()
			if p<=0.5:
				return render(request, 'experiment_docs_init_test.html', {'documents': documents,"query":query_text,"qid":query})
			else:
				return render(request, 'experiment_docs_init_test.html', {'documents': documents,"query":query_text,"qid":query})
    # if a GET (or any other method) we'll create a blank form
	else:
		form = QueryForm()

	return render(request, 'experiment_init.html', {'form': form})

def getQueryInfo(request):
	qid = request.POST.get('action')
	client = MongoClient()
	db = client.asr16
	documents = db.interactive.find({"query_id":qid,"status":"modified"}).sort([("score",-1),("docname",1)])
	query_text = next(db.interactive.find({"query_id":qid,"status":"modified"}))["query"]
	return render(request,'experiment_docs_mod.html',{"documents": documents,"query":query_text,"qid":qid})

def getQueryInfoTest(request):
	qid = request.POST.get('action')
	client = MongoClient()
	db = client.asr16
	documents = db.interactive.find({"query_id":qid,"status":"modified"}).sort([("score",-1),("docname",1)])
	query_text = next(db.interactive.find({"query_id":qid,"status":"modified"}))["query"]
	return render(request,'experiment_docs_mod_test.html',{"documents": documents,"query":query_text,"qid":qid})
	
def finalFormHandle(request):
	group_data = request.POST.get('action')
    # if this is a POST request we need to process the form data
	if request.method == 'POST':
	# create a form instance and populate it with data from the request:
		form = FinalForm(request.POST)
		# check whether it's valid:
		if form.is_valid():
			obj = {}
			obj['name'] = form.cleaned_data['name']
			obj['age'] = form.cleaned_data['age']
			obj['rate_exp'] = form.cleaned_data['rate_exp']
			obj['rate_trust'] = form.cleaned_data['rate_trust']
			obj['rate_stable'] = form.cleaned_data['rate_stable']
			obj['group']=group_data.split("-")[1]
			obj['qid']=group_data.split("-")[0]
			client = MongoClient()
			db = client.asr16
			db.exp_answers.save(obj)
			return render(request, 'thanks.html', {})
	# if a GET (or any other method) we'll create a blank form
	else:
		form = FinalForm()
	return render(request, 'final_form.html', {'form': form, 'group':group_data})
