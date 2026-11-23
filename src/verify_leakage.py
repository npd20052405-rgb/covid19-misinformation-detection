"""Offline, pre-training leakage evidence. Never reads Test or alters raw files."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from difflib import SequenceMatcher
import hashlib
import json
import re
import uuid
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_dataset import normalize

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/constraint2021/030deba36d118c124709fd72d7d5f14c7bc69053'
PREVIOUS = ROOT / 'results/20261018_180718_audit_89987f7d/near_duplicate_candidates.csv'
URL = re.compile(r'https?://\S+|www\.\S+', re.I)


def without_urls(text):
    return re.sub(r'\s+', ' ', URL.sub('', normalize(text))).strip()


def changes(a, b):
    sa, sb = a.split(), b.split()
    return [{'train': ' '.join(sa[i:j]), 'validation': ' '.join(sb[k:l])}
            for tag, i, j, k, l in SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes()
            if tag != 'equal']


def main():
    run = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S') + '_leakage_' + uuid.uuid4().hex[:8]
    out = ROOT / 'results' / run
    out.mkdir()
    provenance = json.loads((RAW / 'provenance.json').read_text(encoding='utf-8'))
    frames = {}
    for split, name in [('train', 'Constraint_Train.csv'), ('validation', 'Constraint_Val.csv')]:
        expected = next(x['sha256'] for x in provenance['files'] if x['name'] == name)
        assert hashlib.sha256((RAW / name).read_bytes()).hexdigest() == expected
        frame = pd.read_csv(RAW / name, dtype=str, encoding='utf-8-sig')
        assert list(frame.columns) == ['id', 'tweet', 'label'] and frame.notna().all().all()
        frame['sample_id'] = split + ':' + frame['id']
        frame['norm'] = frame.tweet.map(normalize)
        frames[split] = frame
    train, val = frames['train'], frames['validation']
    previous = pd.read_csv(PREVIOUS)
    tlookup = train.set_index('sample_id')
    vlookup = val.set_index('sample_id')
    evidence = []
    for row in previous.to_dict('records'):
        a, b = tlookup.loc[row['train_sample_id']], vlookup.loc[row['validation_sample_id']]
        row.update(train_text=a.tweet, validation_text=b.tweet,
                   raw_exact=a.tweet == b.tweet, url_removed_exact=without_urls(a.tweet) == without_urls(b.tweet),
                   changed_spans=changes(normalize(a.tweet), normalize(b.tweet)))
        evidence.append(row)
    with (out / 'original_180_pair_evidence.json').open('x', encoding='utf-8') as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2)
    # Enumerate every cross-split pair above the SAME fixed retrieval threshold.
    vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=40000,
                         min_df=2, dtype=np.float32)
    xt = vec.fit_transform(train.norm)
    xv = vec.transform(val.norm)
    pairs = []
    for start in range(0, len(val), 128):
        similarity = (xv[start:start+128] @ xt.T).tocsr()
        coo = similarity.tocoo()
        for i, j, score in zip(coo.row, coo.col, coo.data):
            if score >= .90:
                a, b = train.iloc[int(j)], val.iloc[start+int(i)]
                pairs.append({'train_sample_id': a.sample_id, 'validation_sample_id': b.sample_id,
                              'cosine': float(score), 'train_label': a.label, 'validation_label': b.label,
                              'raw_exact': a.tweet == b.tweet, 'normalized_exact': a.norm == b.norm,
                              'url_removed_exact': without_urls(a.tweet) == without_urls(b.tweet)})
        print(f'Cross-split screen {min(start+128,len(val))}/{len(val)}', flush=True)
    allpairs = pd.DataFrame(pairs)
    allpairs.to_csv(out / 'all_pairs_ge090.csv', index=False)
    summary = {'run_id': run, 'official_test_read': False, 'training_run': False,
               'method': {'analyzer': 'char', 'ngrams': [3,5], 'max_features': 40000, 'min_df':2,
                          'dtype':'float32','fit_split':'train','threshold':.9,'chunk':128},
               'original_nearest_pairs': len(evidence),
               'original_validation_samples': int(previous.validation_sample_id.nunique()),
               'original_similarity_quantiles': previous.cosine.quantile([0,.25,.5,.75,.9,.95,1]).to_dict(),
               'all_pairs_ge090': len(allpairs),
               'all_pairs_unique_validation': int(allpairs.validation_sample_id.nunique()),
               'all_pairs_unique_train': int(allpairs.train_sample_id.nunique()),
               'all_pairs_different_labels': int((allpairs.train_label != allpairs.validation_label).sum()),
               'raw_exact_pairs': int(allpairs.raw_exact.sum()),
               'url_removed_equal_pairs': int(allpairs.url_removed_exact.sum()),
               'raw_hashes': {x['name']:x['sha256'] for x in provenance['files']}}
    with (out / 'retrieval_summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    fdir = ROOT / 'figures' / run
    fdir.mkdir()
    fig, ax = plt.subplots(figsize=(8,4))
    ax.hist(previous.cosine, bins=np.linspace(.9,1.00001,11), edgecolor='white')
    ax.set(xlabel='Character TF-IDF cosine similarity to nearest Train post',
           ylabel='Unique Validation posts', title='Original 180 candidates: Train–Validation similarity')
    fig.tight_layout(); fig.savefig(fdir / 'candidate_similarity.png',dpi=160); plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    assert without_urls('Hello https://t.co/a') == without_urls('hello https://t.co/b')
    assert without_urls('10 deaths') != without_urls('11 deaths')
    assert without_urls('not safe') != without_urls('safe')
    main()
