from src.jog_memory.jm import JogMemory
from src.jog_memory.rag import JogRag


n_ctx = 2048 + 256
max_tokens = 128 + 128
k=5
n_gpu_layers = 15

jog_memory = JogMemory(
    n_ctx=n_ctx,
    max_tokens=max_tokens,
    n_gpu_layers=n_gpu_layers,
)
jog_rag = JogRag(
    n_ctx=n_ctx,
)

text = """
Ultraviolet A photosensitivity is a debilitating symptom associated with the metabolic disorder Smith-Lemli-Opitz syndrome (SLOS). SLOS is a manifestation of the deficiency of 7-dehydrocholesterol reductase, an enzyme involved in the cholesterol biosynthesis.  As a result several abnormal intermediary compounds are formed among which Cholesta-5,7,9(11)-trien-3β-ol is the most likely cause of photosensitivity. The effect of various drugs acting on cholesterol biosynthetic pathway on SLOS is not clear as clinical trials are not available for this rare disorder. A Flux Balance Analysis (FBA) has been carried out using the software CellNetAnalyzer / FluxAnalyzer to gain insight into the probable effects of various drugs acting on cholesterol biosynthetic pathway on photosensitivity in SLOS. The model consisted of 44 metabolites and 40 reactions. The formation flux of Cholesta-5,7,9(11)-trien-3β-ol increased in SLOS and remained unchanged on simulation of the effect of miconazole and SR31747. However zaragozic acid can potentially reduce the flux through the entire pathway. FBA predicts zaragozic acid along with cholesterol supplementation as an effective treatment for photosensitivity in SLOS.
"""
summary = jog_memory.summarize(text)
print(summary)


