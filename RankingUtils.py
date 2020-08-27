import os
import datetime
import subprocess


def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')


def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)

    out, err = p.communicate()
    return out


def run_svm_rank_model(test_file, model_file, predictions_folder,current_time):
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    predictions_file = predictions_folder + os.path.basename(model_file)+"_"+current_time
    command = "./svm_rank_classify " + test_file + " " + model_file + " " + predictions_file
    print("##Running command: " + command + "##")
    out = run_bash_command(command)
    print("Output of ranking command: " + str(out), flush=True)
    return predictions_file


def run_ranking_model(merged_index, LTR_working_set, current_time, base_dir):
    pathToFolder = base_dir + 'Results/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    INDEX = merged_index
    WORKING_SET_FILE = LTR_working_set
    MODEL_DIR = base_dir + "Code/Models/"
    MODEL_FILE = MODEL_DIR + "experiment_model"
    QUERIES_FILE = base_dir + 'data/queries.xml'
    PREDICTIONS_DIR = "predictions/"
    RANKED_LIST_DIR = "ranked_lists/"
    FEATURES_DIR = pathToFolder + '/Features/' + current_time
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    if not os.path.exists(RANKED_LIST_DIR):
        os.makedirs(RANKED_LIST_DIR)
    ORIGINAL_FEATURES_FILE = 'features'
    command = base_dir + 'Code/LTRFeatures ' + QUERIES_FILE + ' -stream=doc -index=' + INDEX + ' -repository=' + INDEX + ' -useWorkingSet=true -workingSetFile=' + WORKING_SET_FILE + ' -workingSetFormat=trec'
    out = run_bash_command(command)
    print(out)
    run_command('mv doc*_* ' + FEATURES_DIR)
    command = 'perl ' + base_dir + 'Code/generate.pl ' + FEATURES_DIR + ' ' + WORKING_SET_FILE
    out = run_bash_command(command)
    print(out)
    predictions_file = run_svm_rank_model(ORIGINAL_FEATURES_FILE, MODEL_FILE, PREDICTIONS_DIR,current_time)
    command = "perl " + base_dir + 'Code/order.pl ' + RANKED_LIST_DIR + '/SVMRANK' + current_time + ' ' + ORIGINAL_FEATURES_FILE + ' ' + predictions_file
    run_bash_command(command)
    return RANKED_LIST_DIR + '/SVMRANK' + current_time





def run_ranking_model_diff(merged_index, LTR_working_set, current_time, base_dir,new_index_path):
    pathToFolder = base_dir + 'Results/'
    if not os.path.exists(pathToFolder):
        os.makedirs(pathToFolder)
    INDEX = merged_index
    WORKING_SET_FILE = LTR_working_set
    MODEL_DIR = base_dir + "Code/Models/"
    MODEL_FILE = MODEL_DIR + "experiment_model"
    PREDICTIONS_DIR = "predictions/"
    RANKED_LIST_DIR = "ranked_lists/"
    FEATURES_DIR = pathToFolder + '/Features/' + current_time+"/"
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)
    if not os.path.exists(RANKED_LIST_DIR):
        os.makedirs(RANKED_LIST_DIR)
    ORIGINAL_FEATURES_FILE = 'features'
    command="~/jdk1.8.0_181/bin/java -Djava.library.path=/home/greg/indri-5.6/swig/obj/java/ -cp seo_summarization.jar LTRFeatures "+INDEX+" "+new_index_path+" data/stopWordsList data/working_comp_queries.txt "+LTR_working_set+" "+FEATURES_DIR
    out = run_bash_command(command)
    print(out)
    run_command('mv doc*_* ' + FEATURES_DIR)
    command = 'perl ' + base_dir + 'Code/generate.pl ' + FEATURES_DIR + ' ' + WORKING_SET_FILE
    out = run_bash_command(command)
    print(out)
    predictions_file = run_svm_rank_model(ORIGINAL_FEATURES_FILE, MODEL_FILE, PREDICTIONS_DIR,current_time)
    command = "perl " + base_dir + 'Code/order.pl ' + RANKED_LIST_DIR + '/SVMRANK' + current_time + ' ' + ORIGINAL_FEATURES_FILE + ' ' + predictions_file
    run_bash_command(command)
    return RANKED_LIST_DIR + '/SVMRANK' + current_time








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


def create_trectext_for_current_documents(base_dir, texts):
    """
    Create a trec text file of current documents
    """
    path_to_folder = base_dir + 'Collections/'
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    current_time = str(datetime.datetime.now()).replace(":", "-").replace(" ", "-").replace(".", "-")
    path_to_trec_text = path_to_folder + "/TrecText/"
    if not os.path.exists(path_to_trec_text):
        os.makedirs(path_to_trec_text)
    filename = path_to_trec_text + current_time
    query_to_docnos = {}
    f = open(filename, 'w', encoding="utf-8")
    for query in texts:
        query_to_docnos[query] = []
        for document in texts[query]:
            query_to_docnos[query].append(document)
            f.write('<DOC>\n')
            f.write('<DOCNO>' + document + '</DOCNO>\n')
            f.write('<TEXT>\n')
            f.write(str(texts[query][document]).rstrip())
            f.write('\n</TEXT>\n')
            f.write('</DOC>\n')
    f.close()
    path_to_working_set = path_to_folder + '/WorkingSets/'
    if not os.path.exists(path_to_working_set):
        os.makedirs(path_to_working_set)
    working_set_filename = path_to_working_set + current_time
    f = open(working_set_filename, 'w')
    for query, docnos in query_to_docnos.items():
        i = 1
        for docid in docnos:
            f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
            i += 1
    f.close()
    return filename, working_set_filename, current_time


def merge_indices(asr_index, base_dir):
    """
    Merge indices of ASR and ClueWeb09. If MergedIndx is exist, it will be deleted.
    """
    INDRI_DUMP_INDEX = '/home/greg/indri_test/bin/dumpindex'
    CLUEWEB = '/home/greg/cluewebindex'
    path_to_folder = base_dir + 'Collections/'
    MERGED_INDEX = path_to_folder + '/mergedindex'
    run_bash_command('rm -r ' + MERGED_INDEX)
    result = run_bash_command(INDRI_DUMP_INDEX + ' ' + MERGED_INDEX + ' merge ' + CLUEWEB + ' ' + asr_index)
    print("output of merge command = " + str(result))
    return MERGED_INDEX


def build_index(filename, current_time, base_dir):
    """
    Parse the trectext file given, and create an index.
    """
    path_to_folder = base_dir + 'Collections/IndriIndices/'
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)
    INDRI_BUILD_INDEX = '/home/greg/indri_test/bin/IndriBuildIndex'
    CORPUS_PATH = filename
    CORPUS_CLASS = 'trectext'
    MEMORY = '1G'
    INDEX = path_to_folder + current_time
    STEMMER = 'krovetz'
    result = run_bash_command(
        INDRI_BUILD_INDEX + ' -corpus.path=' + CORPUS_PATH + ' -corpus.class=' + CORPUS_CLASS + ' -index=' + INDEX + ' -memory=' + MEMORY + ' -stemmer.name=' + STEMMER)
    print("build index output = " + str(result))
    return INDEX

# baseDir = '/home/greg/ASR18/'
# if not os.path.exists(baseDir):
#     os.makedirs(baseDir)
# trecFileName, workingSetFilename, currentTime = createTrecTextForCurrentDocuments(baseDir)
# LTRWorkingSet, BotWorkingset = createWorkingSetFilesByModels(workingSetFilename)
# mergedIndex = "/home/greg/ASR18/Collections/mergedindex1"
# waterloo_file = "waterloo_scores"
# rankedLists = runRankingModels(mergedIndex, LTRWorkingSet, BotWorkingset, currentTime, baseDir, waterloo_file, workingSetFilename)
