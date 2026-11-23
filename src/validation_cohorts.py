"""Align existing Validation predictions with frozen masks; no fitting or scoring."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def select_validation_cohorts(predictions, review_dir, *, approve_proposed=False):
    review_dir = Path(review_dir)
    protocol = json.loads((review_dir / 'protocol_proposal.json').read_text(encoding='utf-8'))
    mask_file = review_dir / 'validation_evaluation_masks.csv'
    if hashlib.sha256(mask_file.read_bytes()).hexdigest() != protocol['mask_sha256']:
        raise ValueError('Frozen mask hash mismatch')
    masks = pd.read_csv(mask_file)
    if not {'sample_id','true_label'}.issubset(predictions.columns):
        raise ValueError('Need sample_id and true_label')
    if predictions.sample_id.duplicated().any():
        raise ValueError('Duplicate prediction IDs')
    if set(predictions.sample_id) != set(masks.sample_id):
        raise ValueError('Predictions must contain exactly the original Validation IDs; no Train/Test/partial set')
    aligned = predictions.set_index('sample_id').loc[masks.sample_id].reset_index()
    if not aligned.true_label.astype(str).equals(masks.label.astype(str)):
        raise ValueError('Label mismatch after ID alignment')
    names = ['benchmark_original','sensitivity_exact_approved']
    if approve_proposed:
        names.append('sensitivity_verified_proposed')
    return {name: aligned.loc[masks[name]].copy() for name in names}


def self_check():
    root = Path(__file__).resolve().parents[1]
    review_dir = root / 'results/20261018_181449_leakage_11b82aa3/review_v2'
    mask = pd.read_csv(review_dir / 'validation_evaluation_masks.csv')
    # Structural dry-run only: no predicted labels, model scores, or metrics generated.
    fixture = mask[['sample_id','label']].rename(columns={'label':'true_label'}).iloc[::-1].copy()
    groups = select_validation_cohorts(fixture, review_dir)
    assert set(groups) == {'benchmark_original','sensitivity_exact_approved'}
    assert len(groups['benchmark_original']) == 2140
    assert len(groups['sensitivity_exact_approved']) == 2139
    all_groups = select_validation_cohorts(fixture, review_dir, approve_proposed=True)
    assert len(all_groups['sensitivity_verified_proposed']) == 2064
    for invalid in [fixture.iloc[:-1], pd.concat([fixture,fixture.iloc[:1]]),
                    fixture.assign(true_label='invalid'),fixture.assign(sample_id='test:1')]:
        try:
            select_validation_cohorts(invalid,review_dir)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid IDs or labels accepted')
    print('PASS: mask integrity, ID alignment, approval gate, missing/duplicate/Test IDs and label mismatch rejection; no metrics computed')


if __name__ == '__main__':
    self_check()
