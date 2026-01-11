import pickle

with open('/root/ryw/Rec/RLMRec/data/amazon/itm_prf.pkl', 'rb') as f:
    prf = pickle.load(f)

print("type(prf):", type(prf))
print("len(prf):", len(prf))

# 如果是 dict，进一步看结构
if isinstance(prf, dict):
    first_key = next(iter(prf))
    print("key type:", type(first_key))
    print("value type:", type(prf[first_key]))

    v = prf[first_key]
    if hasattr(v, "__len__"):
        print("value length:", len(v))

u = next(iter(prf))
print(prf[u].keys())

for k, v in prf[u].items():
    print(k, type(v))
    if hasattr(v, "__len__"):
        print("  len:", len(v))
