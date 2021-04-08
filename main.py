import json
import pickle
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from typing import Tuple

import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
hostName = "0.0.0.0"
serverPort = 8080

idf: TfidfVectorizer = None
nb_dic : dict = None
rf_dic : dict = None
svm_dic : dict = None


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN


def tokenize_text(original):
    return original.split()


def init_model():
    global idf,nb_dic,svm_dic,rf_dic
    idf = pickle.load(open("./model/tfidf.model", 'rb'))
    nb_dic = pickle.load(open("./model/nb.model", 'rb'))
    svm_dic = pickle.load(open("./model/svm.model", 'rb'))
    rf_dic = pickle.load(open("./model/rf.model", 'rb'))
    print("finished init model")

class MyServer(BaseHTTPRequestHandler):
    def __init__(self, request: bytes, client_address: Tuple[str, int], server: socketserver.BaseServer):
        super().__init__(request, client_address, server)

    def do_GET(self):
        url_to_handler = {
            "/": self.handle_index,

        }
        url_to_handler[self.path]()

    def do_POST(self):
        url_to_handler = {
            "/nb": self.handle_nb,
            "/svm": self.handle_nb,
            "/rf": self.handle_rf,
        }
        url_to_handler[self.path]()

    def handle_index(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        index_file = open("./index.html", "r")
        self.wfile.write(bytes(index_file.read(), 'utf-8'))

    def handle_nb(self):
        content_length = int(self.headers['Content-Length'])
        post_data = str(self.rfile.read(content_length))
        modeled_text = pd.Series(str([tokenize_text(post_data)]))
        modeled_text = idf.transform(modeled_text)
        ret = {}
        for k , v in nb_dic.items():
            ret[k] = float(v.predict(modeled_text))
        j = json.dumps(ret)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        print(j)
        self.wfile.write(bytes(j,'utf-8'))

    def handle_svm(self):
        content_length = int(self.headers['Content-Length'])
        post_data = str(self.rfile.read(content_length))
        modeled_text = pd.Series(str([tokenize_text(post_data)]))
        modeled_text = idf.transform(modeled_text)
        ret = {}
        for k , v in svm_dic.items():
            ret[k] = float(v.predict(modeled_text))
        j = json.dumps(ret)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(j,'utf-8'))

    def handle_rf(self):
        content_length = int(self.headers['Content-Length'])
        post_data = str(self.rfile.read(content_length))
        modeled_text = pd.Series(str([tokenize_text(post_data)]))
        modeled_text = idf.transform(modeled_text)
        ret = {}
        for k , v in rf_dic.items():
            ret[k] = float(v.predict(modeled_text))
        j = json.dumps(ret)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(bytes(j,'utf-8'))



if __name__ == "__main__":
    nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('wordnet')
    init_model()
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))
    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
