"""
Example of training a pre-trained word2vec model on a clinical dataset
* You can also train a model from scratch using the gensim library as shown below:
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
"""
from pypdf import PdfReader
from gensim.models import Word2Vec
from huggingface_hub import snapshot_download

REPO_ID = "garyw/clinical-embeddings-100d-w2v-cr"
FILENAME = "w2v_OA_CR_100d.bin"
word_embedding = Word2Vec.load(snapshot_download(repo_id=REPO_ID)+"/"+FILENAME)
# Read the PDF file
def read_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text = text + " " + page.extract_text()
    return text

# Split text into a list of sentences
def split_text(text):
    return text.replace("\n", "").split(". ")

# Split sentence into a list of words
def split_sentence(sentence):
    # Split the sentence by spaces
    #! Ideally you should split into named entities identified by a NER model
    return sentence.split(" ")

sentences = split_text(read_pdf("examples/sample-soap-note-v4.pdf"))

train_data = []
for i in range(5):
    print(sentences[i])
    print("========\n")
    words = split_sentence(sentences[i])
    train_data.append(words)

word_embedding.train(words, total_examples=len(sentences), epochs=10)

# Save the model
word_embedding.save("data/word_embedding.model")