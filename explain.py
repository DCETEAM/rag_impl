import numpy as np

np.random.seed(7)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def l2_normalize(v):
    n = np.linalg.norm(v)
    return v / n if n != 0 else v

def print_vector(v, label=""):
    tag = f"{label} " if label else ""
    print(f"    {tag}[ {v[0]:+.4f},  {v[1]:+.4f},  {v[2]:+.4f} ]")

def section(title):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)

# ─────────────────────────────────────────────
# STEP 1 — CHUNKING
# ─────────────────────────────────────────────

section("STEP 1 : CHUNKING")

document = (
    "Apple is a technology company. "
    "It develops iPhones MacBooks and software. "
    "AI is becoming important in modern applications."
)

chunks = [
    "Apple is a technology company",
    "It develops iPhones MacBooks and software",
    "AI is becoming important in modern applications",
]

print(f"\n  Document:\n    \"{document}\"\n")
print("  Split into chunks:")
for i, c in enumerate(chunks, 1):
    print(f"\n    Chunk {i} → \"{c}\"")


# ─────────────────────────────────────────────
# STEP 2 — TOKENIZATION  (word-level tokens)
# ─────────────────────────────────────────────

section("STEP 2 : TOKENIZATION — each chunk → tokens")

tokenized = [chunk.split() for chunk in chunks]

for i, (chunk, tokens) in enumerate(zip(chunks, tokenized), 1):
    print(f"\n  Chunk {i}: \"{chunk}\"")
    print(f"  Tokens : {tokens}")
    print(f"  ┌─────┬───────────────────────┐")
    print(f"  │ Pos │ Token                 │")
    print(f"  ├─────┼───────────────────────┤")
    for pos, tok in enumerate(tokens):
        print(f"  │  {pos:<3}│ {tok:<21} │")
    print(f"  └─────┴───────────────────────┘")


# ─────────────────────────────────────────────
# STEP 3 — TOKEN VECTORS  (3D, pre-assigned)
# ─────────────────────────────────────────────

section("STEP 3 : TOKEN VECTORS — each token → 3D vector")

print("""
  Every unique word in our vocabulary is mapped to a 3D vector.
  (In real models these are 384-dim learned embeddings;
   here we use 3D so every number is visible.)

  How to read:  token  →  [ x,  y,  z ]
    x = roughly "tech/product" axis
    y = roughly "action/verb"  axis
    z = roughly "abstract/AI"  axis
""")

# Build a vocab of all unique words across chunks + query
all_words = set()
for t in tokenized:
    all_words.update(t)
query_words = "What products does Apple make".split()
all_words.update(query_words)

# Assign stable 3D vectors per word (seeded random, then normalized)
vocab_vectors = {}
rng = np.random.default_rng(seed=42)
for word in sorted(all_words):
    v = rng.uniform(-1, 1, size=3).astype(np.float32)
    vocab_vectors[word] = l2_normalize(v)

print("  Full Vocabulary → 3D Vectors:")
print(f"  {'Token':<22} {'x':>8}  {'y':>8}  {'z':>8}")
print(f"  {'─'*22} {'─'*8}  {'─'*8}  {'─'*8}")
for word in sorted(vocab_vectors):
    v = vocab_vectors[word]
    print(f"  {word:<22} {v[0]:+.4f}   {v[1]:+.4f}   {v[2]:+.4f}")

print("\n  Token-by-token for each chunk:")
for i, (chunk, tokens) in enumerate(zip(chunks, tokenized), 1):
    print(f"\n  ── Chunk {i}: \"{chunk}\"")
    for tok in tokens:
        v = vocab_vectors[tok]
        print(f"    {tok:<22} →  [ {v[0]:+.4f},  {v[1]:+.4f},  {v[2]:+.4f} ]")


# ─────────────────────────────────────────────
# STEP 4 — CHUNK VECTORS  (mean pooling)
# ─────────────────────────────────────────────

section("STEP 4 : CHUNK VECTORS — token vectors → one doc vector")

print("""
  How?  Mean Pooling:
        chunk_vec = ( token_vec_1 + token_vec_2 + ... + token_vec_N ) / N

  Then L2-Normalize so every vector has length = 1.
""")

chunk_vectors = []

for i, (chunk, tokens) in enumerate(zip(chunks, tokenized), 1):
    token_vecs = np.array([vocab_vectors[t] for t in tokens])

    print(f"  ── Chunk {i}: \"{chunk}\"")
    print(f"\n  Token vectors being summed:")

    running = np.zeros(3, dtype=np.float32)
    for j, (tok, tv) in enumerate(zip(tokens, token_vecs)):
        running = running + tv
        print(f"    + {tok:<20} [ {tv[0]:+.4f},  {tv[1]:+.4f},  {tv[2]:+.4f} ]")
        print(f"      running sum         [ {running[0]:+.4f},  {running[1]:+.4f},  {running[2]:+.4f} ]")

    N = len(tokens)
    raw_mean = running / N
    print(f"\n  Sum / {N} tokens  =  [ {raw_mean[0]:+.4f},  {raw_mean[1]:+.4f},  {raw_mean[2]:+.4f} ]")

    norm_val = np.linalg.norm(raw_mean)
    normalized = raw_mean / norm_val
    print(f"  L2-norm of mean  =  sqrt({raw_mean[0]**2:.4f} + {raw_mean[1]**2:.4f} + {raw_mean[2]**2:.4f})")
    print(f"                   =  sqrt({np.sum(raw_mean**2):.4f})  =  {norm_val:.4f}")
    print(f"\n  Normalized chunk vector  =  mean / {norm_val:.4f}")
    print(f"  ✦ Chunk {i} Vector  →  [ {normalized[0]:+.4f},  {normalized[1]:+.4f},  {normalized[2]:+.4f} ]")
    print()

    chunk_vectors.append(normalized)

chunk_vectors = np.array(chunk_vectors)

print("  ── All Chunk Vectors (final):")
for i, cv in enumerate(chunk_vectors, 1):
    print(f"    Chunk {i}  →  [ {cv[0]:+.4f},  {cv[1]:+.4f},  {cv[2]:+.4f} ]")


# ─────────────────────────────────────────────
# STEP 5 — QUERY VECTOR
# ─────────────────────────────────────────────

section("STEP 5 : QUERY VECTOR — same process as chunk")

query = "What products does Apple make"
query_tokens = query.split()

print(f"\n  Query: \"{query}\"")
print(f"  Tokens: {query_tokens}")

print(f"\n  Token vectors:")
running_q = np.zeros(3, dtype=np.float32)
for tok in query_tokens:
    tv = vocab_vectors[tok]
    running_q = running_q + tv
    print(f"    + {tok:<20} [ {tv[0]:+.4f},  {tv[1]:+.4f},  {tv[2]:+.4f} ]")
    print(f"      running sum         [ {running_q[0]:+.4f},  {running_q[1]:+.4f},  {running_q[2]:+.4f} ]")

N_q = len(query_tokens)
raw_mean_q = running_q / N_q
norm_q = np.linalg.norm(raw_mean_q)
query_vector = raw_mean_q / norm_q

print(f"\n  Sum / {N_q}  =  [ {raw_mean_q[0]:+.4f},  {raw_mean_q[1]:+.4f},  {raw_mean_q[2]:+.4f} ]")
print(f"  L2-norm   =  sqrt({raw_mean_q[0]**2:.4f} + {raw_mean_q[1]**2:.4f} + {raw_mean_q[2]**2:.4f})")
print(f"            =  sqrt({np.sum(raw_mean_q**2):.4f})  =  {norm_q:.4f}")
print(f"\n  ✦ Query Vector  →  [ {query_vector[0]:+.4f},  {query_vector[1]:+.4f},  {query_vector[2]:+.4f} ]")


# ─────────────────────────────────────────────
# STEP 6 — COSINE SIMILARITY
# ─────────────────────────────────────────────

section("STEP 6 : COSINE SIMILARITY — query vs each chunk")

print("""
  Formula:
        cosine_similarity(Q, C)  =   dot(Q, C)
                                   ─────────────────
                                    ||Q||  ×  ||C||

  Since all vectors are L2-normalized, ||Q|| = ||C|| = 1
  So:   cosine_similarity(Q, C)  =   dot(Q, C)
                                   =   Q[0]*C[0] + Q[1]*C[1] + Q[2]*C[2]
""")

Q = query_vector
scores = []

for i, (chunk, C) in enumerate(zip(chunks, chunk_vectors), 1):
    print(f"  ── Query vs Chunk {i}: \"{chunk}\"")
    print(f"\n    Q  =  [ {Q[0]:+.4f},  {Q[1]:+.4f},  {Q[2]:+.4f} ]")
    print(f"    C{i}  =  [ {C[0]:+.4f},  {C[1]:+.4f},  {C[2]:+.4f} ]")

    t0 = Q[0] * C[0]
    t1 = Q[1] * C[1]
    t2 = Q[2] * C[2]
    dot = t0 + t1 + t2

    print(f"\n    dot(Q, C{i}):")
    print(f"      Q[0] × C[0]  =  {Q[0]:+.4f} × {C[0]:+.4f}  =  {t0:+.6f}")
    print(f"      Q[1] × C[1]  =  {Q[1]:+.4f} × {C[1]:+.4f}  =  {t1:+.6f}")
    print(f"      Q[2] × C[2]  =  {Q[2]:+.4f} × {C[2]:+.4f}  =  {t2:+.6f}")
    print(f"                                          ─────────────")
    print(f"      Sum          =  {t0:+.6f} + {t1:+.6f} + {t2:+.6f}  =  {dot:+.6f}")

    mag_Q = np.sqrt(Q[0]**2 + Q[1]**2 + Q[2]**2)
    mag_C = np.sqrt(C[0]**2 + C[1]**2 + C[2]**2)
    print(f"\n    ||Q||  =  sqrt({Q[0]**2:.4f} + {Q[1]**2:.4f} + {Q[2]**2:.4f})  =  {mag_Q:.4f}")
    print(f"    ||C{i}||  =  sqrt({C[0]**2:.4f} + {C[1]**2:.4f} + {C[2]**2:.4f})  =  {mag_C:.4f}")

    similarity = dot / (mag_Q * mag_C)
    print(f"\n    cosine_sim  =  {dot:+.6f} / ({mag_Q:.4f} × {mag_C:.4f})")
    print(f"                =  {dot:+.6f} / {mag_Q * mag_C:.4f}")
    print(f"    ✦ Score     =  {similarity:+.6f}")
    scores.append((similarity, i, chunk))
    print()


# ─────────────────────────────────────────────
# STEP 7 — RESULT
# ─────────────────────────────────────────────

section("STEP 7 : RESULT — ranked matches")

scores.sort(reverse=True)

print(f"\n  Query: \"{query}\"\n")
print(f"  {'Rank':<6} {'Score':>10}   Chunk")
print(f"  {'─'*4}  {'─'*10}   {'─'*44}")
for rank, (score, idx, chunk) in enumerate(scores, 1):
    marker = " ◄ BEST MATCH" if rank == 1 else ""
    print(f"  #{rank:<5} {score:>+10.6f}   Chunk {idx}: \"{chunk}\"{marker}")

print(f"\n  ✦ The chunk most similar to the query is:")
print(f"    \"{scores[0][2]}\"")
print(f"\n  This gets sent to the LLM as context to answer the query.\n")