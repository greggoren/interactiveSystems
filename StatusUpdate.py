from pymongo import MongoClient

client = MongoClient('asr2.iem.technion.ac.il', 27017)
db = client.asr16
docs = db.interactive.find({})
for doc in docs:
    doc["status"]="init"
    db.interactive.save(doc)