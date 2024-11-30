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

# For each sentence in the text
def split_text(text):
    return text.replace("\n", "").split(". ")

# Split sentence into a list of words
def split_sentence(sentence):
    return sentence.split(" ")

sentences = split_text(read_pdf("examples/sample-soap-note-v4.pdf"))

# Print the first 5 sentences
train_data = []
for i in range(5):
    print(sentences[i])
    print("========\n")
    words = split_sentence(sentences[i])
    train_data.append(words)

word_embedding.train(words, total_examples=len(sentences), epochs=10)

# Save the model
word_embedding.save("data/word_embedding.model")