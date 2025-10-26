# decision/nlp/rule_parser.py
from sentence_transformers import SentenceTransformer, util
import numpy as np

MODEL = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast

# Predefined keywords mapping
KEYWORD_MAP = {
    'blur all': {'mode':'all'},
    'blur everyone': {'mode':'all'},
    'blur students': {'target':'student'},
    'blur children': {'target':'child'},
    'keep teacher': {'preserve':'teacher'},
    'blur background': {'target':'background'},
    'blur non-speaker': {'target':'non_speaker'},
    'blur crowd': {'mode':'crowd'},
}

# Embedding templates for semantic lookup
TEMPLATES = list(KEYWORD_MAP.keys())
TEMPLATES_EMB = MODEL.encode(TEMPLATES, convert_to_tensor=True)

def parse_user_text(text, threshold=0.55):
    text = text.strip().lower()
    # direct keyword match
    for k in KEYWORD_MAP:
        if k in text:
            out = KEYWORD_MAP[k].copy()
            out['raw'] = text
            return out
    # semantic fallback
    emb = MODEL.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(emb, TEMPLATES_EMB)[0].cpu().numpy()
    idx = int(np.argmax(scores))
    if scores[idx] >= threshold:
        out = KEYWORD_MAP[TEMPLATES[idx]].copy()
        out['raw'] = text
        return out
    # default: return parsed intent = none
    return {'raw': text, 'mode': None}
