# https://graphsinspace.net
# https://tigraphs.pmf.uns.ac.rs

# GRASP, deterministički

"""
Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

mašinsko učenje na grafovima 
- graph embeddings (ugradnje grafova / čvorova u vektorske prostore)
trening embedding cvorova na nekom grafu

složeni podaci koji prirodno imaju strukturu grafa  
predstave se kao vektori tako da:
slični čvorovi/grafovi budu blizu u prostoru 

GRASP razvija i primenjuje metode: 
Node2Vec, DeepWalk, GraphSAGE, GCN, GAT 
"""

import numpy as np
import pandas as pd


df = pd.read_csv('/data/loto7hh_4580_k21.csv')
print()
print(df)
print()
"""
      Num1  Num2  Num3  Num4  Num5  Num6  Num7
0        5    14    15    17    28    30    34
1        2     3    13    18    19    23    37
2       13    17    18    20    21    26    39
3       17    20    23    26    35    36    38
4        3     4     8    11    29    32    37
...    ...   ...   ...   ...   ...   ...   ...
4575     2     5    12    14    23    24    32
4576     5     6    12    18    22    29    31
4577     1     9    10    16    22    24    31
4578    13    16    24    28    33    34    35
4579     9    10    27    29    30    34    37

[4580 rows x 7 columns]
"""


from itertools import combinations

CSV_PATH = "/data/loto7hh_4580_k21.csv"

SEED = 39
np.random.seed(SEED)

W_FREQ = 1.0
W_PAIR = 1.0


def load_draws(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path, encoding="utf-8")
    expected_cols = [f"Num{i}" for i in range(1, 8)]
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Nedostaje kolona {c} u CSV fajlu.")
    draws = []
    for _, row in df.iterrows():
        nums = [int(row[f"Num{i}"]) for i in range(1, 8)]
        nums_sorted = sorted(nums)
        draws.append(nums_sorted)
    return draws


def compute_frequencies(draws):
    counts = {i: 0 for i in range(1, 40)}
    for draw in draws:
        for n in draw:
            counts[n] += 1
    total = sum(counts.values())
    freqs = {i: counts[i] / total for i in counts}
    return counts, freqs


def compute_cooccurrence_matrix(draws):
    # M[i,j] = koliko puta su se brojevi i i j pojavili zajedno u istom izvlačenju
    M = np.zeros((40, 40), dtype=np.int64)  # indeksiramo 1..39, red 0/kol 0 ne koristimo
    for draw in draws:
        for i_idx in range(len(draw)):
            for j_idx in range(i_idx + 1, len(draw)):
                a = draw[i_idx]
                b = draw[j_idx]
                M[a, b] += 1
                M[b, a] += 1
    return M


def compute_spectral_embeddings(M, k=7):
    # Gradimo Laplasijan grafa na osnovu M
    degrees = M.sum(axis=1)
    L = np.diag(degrees) - M
    # Uklanjamo red/kolonu 0 (ne koristimo broj 0)
    L_sub = L[1:40, 1:40]
    # Eigen-dekompozicija (realno simetrična matrica)
    vals, vecs = np.linalg.eigh(L_sub)
    # Preskačemo prvi eigenvektor, uzimamo narednih k
    if k + 1 > vecs.shape[1]:
        k = vecs.shape[1] - 1
    embedding = vecs[:, 1:1 + k]
    return embedding  # shape (39, k), red i-1 = broj i


def score_combo(combo, counts, M):
    # combo = tuple/list od 7 različitih brojeva (sortiranih)
    # frekvencijski deo
    freq_part = sum(counts[i] for i in combo)
    # deo sa ko-pojavnostima
    pair_part = 0
    for i, j in combinations(combo, 2):
        pair_part += M[i, j]
    score = W_FREQ * freq_part + W_PAIR * pair_part
    return score


def find_best_combo(counts, M):
    best_score = None
    best_combo = None

    # prolaz kroz SVE kombinacije 7 od 39 (deterministički, bez random)
    for combo in combinations(range(1, 40), 7):
        s = score_combo(combo, counts, M)
        if best_score is None or s > best_score:
            best_score = s
            best_combo = combo

    return best_combo, best_score


def main():
    draws = load_draws()
    print()
    print(f"Ukupno izvlačenja: {len(draws)}")
    print()
    """
    Ukupno izvlačenja: 4580
    """
    
    counts, freqs = compute_frequencies(draws)
    print()
    print("Frekvencije brojeva 1-39:")
    print()
    for i in range(1, 40):
        print(f"{i:2d}: count={counts[i]:4d}, freq={freqs[i]:.6f}")
    print()
    """
    Frekvencije brojeva 1-39:

    1: count= 773, freq=0.024111
    2: count= 817, freq=0.025483
    3: count= 820, freq=0.025577
    
    8: count= 901, freq=0.028104
    
    23: count= 898, freq=0.028010
    
    37: count= 853, freq=0.026606
    38: count= 828, freq=0.025827
    39: count= 838, freq=0.026138
    """

    
    M = compute_cooccurrence_matrix(draws)
    print()
    print("Dimenzija matrice sa-pojavljivanja:", M.shape)
    print()
    """
    Dimenzija matrice sa-pojavljivanja: (40, 40)
    """


    emb = compute_spectral_embeddings(M, k=7)
    print()
    print("Dimenzija embedding matrice:", emb.shape)
    print()
    """
    Dimenzija embedding matrice: (39, 7)
    """

    print()
    print()
    print("Prvih nekoliko embedding vektora (brojevi 1-10):")
    print()
    for i in range(1, 11):
        vec = emb[i - 1]
        vec_str = ", ".join(f"{x:.6f}" for x in vec)
        print(f"{i:2d}: [{vec_str}]")
    print()
    print()
    """
    Prvih nekoliko embedding vektora (brojevi 1-10):

    1: [0.039687, -0.346944, 0.665814, -0.243481, 0.096534, -0.549364, -0.037322]
    2: [-0.011911, -0.050331, 0.047436, 0.015633, -0.031961, -0.010311, 0.077360]
    3: [-0.003705, -0.019123, 0.016759, 0.070903, 0.062146, 0.068988, -0.000805]
    4: [-0.012751, -0.076497, 0.009056, 0.177419, -0.054684, 0.069683, 0.046801]
    5: [0.017092, -0.056673, -0.007424, 0.062233, 0.013653, 0.079000, 0.035340]
    6: [0.011646, -0.073355, -0.060742, -0.074303, -0.071860, 0.014996, 0.184158]
    7: [0.005636, -0.038984, 0.036661, 0.040922, 0.013307, 0.008600, 0.048885]
    8: [0.014249, -0.025986, -0.002530, 0.001766, 0.002512, 0.018894, -0.000352]
    9: [0.027825, -0.010970, -0.028272, 0.059429, 0.000701, 0.048025, 0.046890]
    10: [0.022925, -0.020019, -0.000097, -0.016288, 0.023554, 0.071429, 0.019955]
    """

    best_combo, best_score = find_best_combo(counts, M)

    print()
    print("\nPredikcija sledeće kombinacije (deterministički):")
    print("Kombinacija:", best_combo)
    print("Score:", best_score)
    print()
    """
    Predikcija sledeće kombinacije (deterministički):
    Kombinacija: (8, 11, x, y, z, 33, 34)
    Score: 9230.0
    """


if __name__ == "__main__":
    main()
