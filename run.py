import os
import timeit
from apriori import Apriori

PATH = os.getcwd() + "/test/data/T10I4D100K.dat"
N = 100000
SUPPORT = 0.01
CONFIDENCE = 0.20

# create a new instance of Apriori
apriori = Apriori()

# load baskets
lb_ms = timeit.timeit("apriori.load_baskets(PATH, n=N, sep=' ', duplicates=False, verbose=True)", number=1, globals=globals())
print("Baskets loaded: {} seconds".format(lb_ms))

for basket in apriori.baskets[:10]:
    print(basket)

# get candidates and frequent itemsets
c_ms = timeit.timeit("apriori.compute(min_support=SUPPORT, stop=False, verbose=False)", number=1, globals=globals())
print("Apriori computed: {} seconds".format(c_ms))

for i in range(0, len(apriori.candidates)):
    print("C{}: {} candidates".format(i, len(apriori.candidates[i])))
    print("L{}: {} frequent itemsets".format(i, len(apriori.frequent_itemsets[i])))

gar_ms = timeit.timeit("apriori.get_association_rules(min_confidence=CONFIDENCE, verbose=True)", number=1, globals=globals())
print("Association rules computed: {} seconds".format(gar_ms))

print("Association rules: {} rules".format(len(apriori.association_rules)))
for r, (c, s) in apriori.association_rules.items():
    print("Rule: {} -> {} - Confidence: {} - Support: {}".format(list(r[0]), list(r[1]), c, s))

# COMPARISON with other libraries

# custom rules
custom_rules = [(r[0], r[1]) for r in apriori.association_rules.keys()]

# Prepare dataset
dataset = []
with open(PATH) as f:
    for line in f:
        dataset.append(line.rstrip().split(' '))

# mlx apriori

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori as apriori_mlx
from mlxtend.frequent_patterns import association_rules

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
fi = apriori_mlx(df, min_support=SUPPORT, use_colnames=True)
ar = association_rules(fi, metric="confidence", min_threshold=CONFIDENCE)
mlxtend_rules = [(row['antecedents'], row['consequents']) for index, row in ar.iterrows()]
print("MLXTEND: {} rules".format(len(mlxtend_rules)))

# apyori

from apyori import apriori as apriori_ap

apyori_rules = list(apriori_ap(dataset, min_support=SUPPORT, min_confidence=CONFIDENCE))
print("APYORI: {} rules".format(len(apyori_rules)))

# efficient-apriori

from efficient_apriori import apriori as apriori_ef

itemsets, ef_rules = apriori_ef(dataset, min_support=SUPPORT,  min_confidence=CONFIDENCE)
print("EFFICIENT-APRIORI: {} rules".format(len(ef_rules)))

print("Rules: Mlx {} - Apyori {} - Efficient {} - Custom {}".format(len(mlxtend_rules), len(apyori_rules), len(ef_rules), len(custom_rules)))

# VISUALIZATION

# scatterplot

if apriori.association_rules:
    df = pd.DataFrame(apriori.association_rules.values())
    df.columns = ["confidence", "support"]
    df.plot.scatter(x="confidence", y="support")
else:
    print("No association rules")

# connected graph

import networkx as nx
import matplotlib.pyplot as plt

if apriori.association_rules:
    plt.figure(figsize=(10, 10))
    G = nx.DiGraph()
    for r, (c, s) in apriori.association_rules.items():
        end = list(r[1])[0]
        for rx in list(r[0]):
            G.add_edge(rx, end, weight=1, arrowsize=100)
    edges = [
        (u, v) for (u, v, d) in G.edges(data=True)
    ]
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    plt.axis("off")
    plt.show()
    print("")
else:
    print("No association rules")