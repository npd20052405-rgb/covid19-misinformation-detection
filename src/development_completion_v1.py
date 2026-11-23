"""Visible notebook stages 09–11; import has no I/O or fitting. Test remains locked."""
from pathlib import Path
from datetime import datetime
from itertools import combinations
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import uuid
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
import pilot_h1_h2 as pilot

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/20261019_development_completion_v1'
CFG_PATH = ROOT / 'configs/development_completion_v1.json'
OLD = ROOT / 'results/20261018_h1_h2_pilot_seed42_01'
B_RUN = ROOT / 'results/20261019_direction_B_seed42_01'
REVIEW = ROOT / 'results/20261018_181449_leakage_11b82aa3/review_v2'
MEMBERS = {'B0': ['word_nb', 'char_lr'], 'B1': ['word_svm', 'char_lr'],
           'B2': ['word_nb', 'char_lr', 'word_svm']}
CLASSES = np.array(['fake', 'real'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(relative):
    path = (ROOT / relative).resolve()
    assert path.is_relative_to(ROOT) and 'test' not in str(path.relative_to(ROOT)).lower()
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, frame):
    frame.to_csv(path, index=False, mode='x')


def check_map(mapping):
    for relative, expected in mapping.items():
        assert sha(safe(relative)) == expected, 'Checksum đổi: ' + relative


def old_stage(run, name):
    marker = read_json(run / (name + '_COMPLETE.json'))
    check_map(marker['files'])
    return safe(marker['folder'])


def completed(name):
    path = OUT / (name + '_COMPLETE.json')
    if not path.exists():
        return None
    marker = read_json(path)
    check_map(marker['files'])
    return safe(marker['folder'])


def attempt(name):
    folder = OUT / (name + '_attempt_' + uuid.uuid4().hex[:8])
    folder.mkdir(exist_ok=False)
    return folder


def commit(name, folder, seconds):
    files = {str(p.relative_to(ROOT)): sha(p) for p in folder.rglob('*') if p.is_file()}
    write_json(OUT / (name + '_COMPLETE.json'), {'folder': str(folder.relative_to(ROOT)),
               'files': files, 'seconds': seconds, 'time': datetime.now().astimezone().isoformat()})
    with (OUT / 'run.log').open('a', encoding='utf-8') as f:
        f.write(datetime.now().astimezone().isoformat() + ' ' + name + ' COMPLETE\n')
    # Progress is append-only: interruptions do not erase an earlier state.
    write_json(OUT / ('state_' + name + '.json'), {'last_completed': name,
               'test_read': False, 'next': 'continue planned stages only; final report then stop'})


def guard():
    prepared = completed('00_preflight')
    assert prepared is not None, 'Chạy preflight trước.'
    lock = read_json(prepared / 'protocol_lock.json')
    check_map(lock['input_hashes'])
    check_map(lock['code_config_hashes'])
    assert read_json(CFG_PATH) == lock['config']
    return prepared, lock['config']


def preflight():
    """Validate cached stages; do not rerun any notebook 00–08."""
    if completed('00_preflight') is not None:
        guard()
        return pd.read_csv(completed('00_preflight') / 'split_summary.csv')
    assert not OUT.exists(), 'Run đã tồn tại nhưng preflight chưa hoàn tất; giữ lại để kiểm tra.'
    OUT.mkdir()
    start = time.perf_counter()
    folder = attempt('00_preflight')
    cfg = read_json(CFG_PATH)
    assert cfg['seed_base_assignment'] == 42 and not cfg['official_test_access']
    assert cfg['new_cv_folds'] == 0 and cfg['new_fit_budget'] == 18
    self_check()
    hashes = {}
    for run in [OLD, B_RUN]:
        for path in sorted(run.glob('*_COMPLETE.json')):
            marker = read_json(path)
            check_map(marker['files'])
            hashes[str(path.relative_to(ROOT))] = sha(path)
            hashes.update(marker['files'])
    hashes.update(read_json(B_RUN / 'inputs.json'))
    # Read text/source histories without touching any Test file.
    inventory = []
    for directory in ['docs', 'prompts', 'reports', 'notebooks']:
        for path in sorted((ROOT / directory).glob('*')):
            if path.is_file() and path.suffix in ['.md', '.ipynb'] and 'test' not in path.name.lower():
                content = path.read_text(encoding='utf-8-sig')
                row = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'characters': len(content)}
                if path.suffix == '.ipynb':
                    nb = json.loads(content)
                    code = [c for c in nb['cells'] if c['cell_type'] == 'code']
                    row.update(code_cells=len(code), executed_cells=sum(c.get('execution_count') is not None for c in code),
                               errors=sum(o.get('output_type') == 'error' for c in code for o in c.get('outputs', [])))
                inventory.append(row)
                if directory != 'notebooks' or path.name[:2].isdigit() and int(path.name[:2]) <= 8:
                    hashes[str(path.relative_to(ROOT))] = sha(path)
    for rel in ['AGENTS.md', 'README.md',
                'reports/20261019_h1_h2_final_synthesis/validation_dependency_proxy.csv',
                'reports/20261019_h1_h2_final_synthesis/dependency_group_summary.csv',
                'results/20261018_181449_leakage_11b82aa3/review_v2/validation_evaluation_masks.csv',
                'results/20261018_190510_preapproval_f939ffb0/metrics_all.json',
                'results/20261018_190510_preapproval_f939ffb0/inference_timing.csv',
                'results/20261018_183010_bert_tiny_8945ea6b/history.csv']:
        hashes[rel] = sha(safe(rel))
    check_map(hashes)
    prepared = old_stage(OLD, '00_prepare')
    assignment = pd.read_csv(prepared / 'assignment.csv')
    assert len(assignment) == 6420 and assignment.sample_id.is_unique
    assert assignment.groupby('group_id').role.nunique().eq(1).all()
    assert assignment.role.value_counts().to_dict() == {'A': 4494, 'C': 642, 'M': 642, 'H': 642}
    assert sha(prepared / 'assignment.csv') == 'cb457e35c0eb68fc8fb95ec8096f0a8db11d0ae4a9c8a7f25a93d77479e8a7b9'
    code_hashes = {str(Path(__file__).relative_to(ROOT)): sha(__file__),
                   str(CFG_PATH.relative_to(ROOT)): sha(CFG_PATH)}
    for name in ['09_Development_Paired_Bootstrap.ipynb', '10_Development_Stability_Ablation.ipynb',
                 '11_Development_Final_Dossier.ipynb']:
        path = ROOT / 'notebooks' / name
        nb = read_json(path)
        # Code-only snapshot: execution counts and rendered outputs may change.
        write_json(folder / (name + '.code.json'), [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'])
    write_json(folder / 'protocol_lock.json', {'config': cfg, 'input_hashes': hashes,
               'code_config_hashes': code_hashes, 'selection': None,
               'status': 'LOCKED_BEFORE_NEW_COMBINER_DIAGNOSTIC_FITS_NOT_FINAL_MODEL'})
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'joblib', 'threadpoolctl']
    resource = {'python': platform.python_version(), 'cpu_count': os.cpu_count(),
                'versions': {p: importlib.metadata.version(p) for p in packages},
                'threads': 2, 'GPU_used': False, 'base_fit': False, 'encoder_added': False}
    previous_env = read_json(prepared / 'environment.json')
    assert resource['python'] == previous_env['python'] and resource['versions'] == previous_env['versions']
    try:
        import psutil
        resource.update(ram_total_GB=psutil.virtual_memory().total / 1024**3,
                        ram_available_GB=psutil.virtual_memory().available / 1024**3)
    except ImportError:
        resource['ram'] = 'N/A; psutil unavailable; no installation'
    write_json(folder / 'resource.json', resource)
    write_csv(folder / 'inventory.csv', pd.DataFrame(inventory))
    summary = pd.crosstab(assignment.role, assignment.label).reindex(['A', 'C', 'M', 'H']).reset_index()
    write_csv(folder / 'split_summary.csv', summary)
    commit('00_preflight', folder, time.perf_counter() - start)
    print('PASS: kế thừa 00–08; protocol khóa; Test không đọc; không chạy lại base.', flush=True)
    return summary


def frame(ref, q):
    assert q.shape == (len(ref), 2) and np.isfinite(q).all()
    assert (q >= 0).all() and np.allclose(q.sum(axis=1), 1)
    result = ref[['sample_id', 'true_label']].copy()
    result['predicted_label'] = CLASSES[q.argmax(axis=1)]
    result['prob_fake'], result['prob_real'] = q[:, 0], q[:, 1]
    return result


def align(frames, names):
    ref = frames[names[0]][['sample_id', 'true_label']].copy()
    blocks = []
    for name in names:
        x = frames[name]
        assert x.sample_id.is_unique and set(x.sample_id) == set(ref.sample_id)
        x = x.set_index('sample_id').loc[ref.sample_id].reset_index()
        assert np.array_equal(ref.true_label, x.true_label)
        q = x[['prob_fake', 'prob_real']].to_numpy()
        assert np.isfinite(q).all() and (q >= 0).all() and np.allclose(q.sum(axis=1), 1)
        blocks.append(q)
    return ref, np.stack(blocks, axis=1)


def pooling(p, w, pairs=False, return_scores=False):
    k = p.shape[1]
    basis = [p[:, i, :] for i in range(k)]
    if pairs:
        basis += [np.minimum(p[:, i, :], p[:, j, :]) for i, j in combinations(range(k), 2)]
    w = np.asarray(w)
    assert len(w) == len(basis) and np.isfinite(w).all() and w.min() >= -1e-12
    scores = np.einsum('nkc,k->nc', np.stack(basis, axis=1), w)
    if return_scores:
        return scores
    total = scores.sum(axis=1, keepdims=True)
    return np.divide(scores, total, out=np.full_like(scores, .5), where=total > 1e-15)


def cached_cohorts():
    result = {}
    for stage_name, cohort in [('02_H', 'H_exploratory'), ('03_Validation', 'Validation')]:
        folder = old_stage(B_RUN, stage_name)
        cache = {p.stem.removesuffix('_predictions'): pd.read_csv(p)
                 for p in sorted(folder.glob('*_predictions.csv'))}
        raw = old_stage(OLD, '05_audit_H' if stage_name == '02_H' else '05_audit_Validation')
        cache['BASE__char_lr_raw'] = pd.read_csv(raw / 'char_lr_raw_predictions.csv')
        if cohort != 'Validation':
            result[cohort] = cache
        else:
            for key, f in cache.items():
                for name, subset in pilot.select_validation_cohorts(f, REVIEW, approve_proposed=True).items():
                    result.setdefault(name, {})[key] = subset
    return result


def comparison_list():
    comps = []
    for c in MEMBERS:
        names = ['weighted_mean', 'equal_mean'] + (['normalized_pair_min'] if c != 'B2' else [])
        comps += [('operator', c + '__learned_choquet', c + '__' + n) for n in names]
    for a, b in [('B1', 'B0'), ('B2', 'B0'), ('B2', 'B1')]:
        names = ['equal_mean', 'weighted_mean', 'learned_choquet'] + (['normalized_pair_min'] if a == 'B1' else [])
        comps += [('members', a + '__' + n, b + '__' + n) for n in names]
    ensemble_names = [c + '__' + n for c in MEMBERS for n in
                      ['equal_mean', 'weighted_mean', 'learned_choquet'] + (['normalized_pair_min'] if c != 'B2' else [])]
    comps += [('vs_single', a, b) for a in ensemble_names for b in
              ['BASE__word_svm', 'BASE__char_lr', 'BASE__char_lr_raw']]
    comps += [('cross_control', 'B2__learned_choquet', 'B1__normalized_pair_min')]
    assert len(comps) == len(set((a, b) for _, a, b in comps)) == 52
    return comps


def self_check():
    p = np.array([[[.8, .2], [.3, .7]], [[1., 0.], [0., 1.]], [[.1, .9], [.4, .6]]])
    assert np.allclose(pooling(p, [.5, .5]), p.mean(axis=1))
    assert np.allclose(pooling(p, [0., 0., 1.], True), pilot.pool(p, [0., 0., 1.]))
    q = np.concatenate([p, p[:, :1]], axis=1)
    assert np.allclose(pooling(q, [1/3]*3+[0.]*3, True), q.mean(axis=1))
    assert np.allclose(pooling(p, [0., 0., 1.], True)[1], [.5, .5])
    ref = pd.DataFrame({'sample_id': ['a', 'b', 'c'], 'true_label': ['fake', 'real', 'real']})
    f = frame(ref, pooling(p, [.5, .5]))
    scores, _ = pilot.evaluate(f)
    y = (f.true_label == 'real').to_numpy(dtype=int)
    pred = (f.predicted_label == 'real').to_numpy(dtype=int)
    features = np.zeros((3, 6))
    features[np.arange(3), 2*y+pred] = 1
    proba = f[['prob_fake', 'prob_real']].to_numpy()
    features[:, 4] = -np.log(proba[np.arange(3), y])
    features[:, 5] = (proba[:, 1]-y)**2
    actual = metric_draws(features.sum(axis=0)[None, :])[0]
    assert np.allclose(actual, [scores['weighted_f1'], scores['log_loss'], scores['brier_real_binary']], atol=1e-12)
    assert len(comparison_list()) == 52


def metric_draws(s):
    c = s[..., :4]
    n = c.sum(axis=-1)
    sup0, sup1 = c[..., 0] + c[..., 1], c[..., 2] + c[..., 3]
    den0, den1 = 2*c[..., 0] + c[..., 1] + c[..., 2], 2*c[..., 3] + c[..., 1] + c[..., 2]
    f0 = np.divide(2*c[..., 0], den0, out=np.zeros_like(n), where=den0 > 0)
    f1 = np.divide(2*c[..., 3], den1, out=np.zeros_like(n), where=den1 > 0)
    return np.stack([(sup0*f0 + sup1*f1)/n, s[..., 4]/n, s[..., 5]/n], axis=-1)


def bootstrap():
    _, cfg = guard()
    done = completed('01_bootstrap')
    if done is not None:
        return pd.read_csv(done / 'paired_ci.csv')
    start = time.perf_counter()
    folder = attempt('01_bootstrap')
    cohorts = cached_cohorts()
    assignment = pd.read_csv(old_stage(OLD, '00_prepare') / 'assignment.csv').set_index('sample_id')
    proxy = pd.read_csv(ROOT / 'reports/20261019_h1_h2_final_synthesis/validation_dependency_proxy.csv').set_index('sample_id')
    comps = comparison_list()
    write_json(folder / 'comparisons_before_resampling.json', comps)
    rows, errors, groups, draws = [], [], [], {}
    for ci, (cohort, cache) in enumerate(cohorts.items()):
        keys = sorted(cache)
        index = {k: i for i, k in enumerate(keys)}
        reference = cache[keys[0]].set_index('sample_id')
        ids = reference.index
        y = (reference.true_label == 'real').to_numpy(dtype=int)
        features = np.zeros((len(ids), len(keys), 6))
        for k, name in enumerate(keys):
            f = cache[name].set_index('sample_id').loc[ids]
            assert np.array_equal(f.true_label, reference.true_label)
            p = f[['prob_fake', 'prob_real']].to_numpy()
            assert np.array_equal(f.predicted_label, CLASSES[p.argmax(axis=1)])
            predicted = (f.predicted_label == 'real').to_numpy(dtype=int)
            features[np.arange(len(ids)), k, 2*y + predicted] = 1
            features[:, k, 4] = -np.log(np.clip(p[np.arange(len(ids)), y], np.finfo(float).eps, 1))
            features[:, k, 5] = (p[:, 1] - y)**2
            score, _ = pilot.evaluate(f.reset_index())
            actual = metric_draws(features[:, k].sum(axis=0)[None, :])[0]
            assert np.allclose(actual, [score['weighted_f1'], score['log_loss'], score['brier_real_binary']], atol=1e-12, rtol=0)
        point = metric_draws(features.sum(axis=0))
        gsource = assignment.loc[ids].group_id.to_numpy() if cohort == 'H_exploratory' else proxy.loc[ids].dependency_proxy_group.to_numpy()
        _, group_idx = np.unique(gsource, return_inverse=True)
        counts = np.bincount(group_idx)
        groups.append({'cohort': cohort, 'n': len(ids), 'groups': len(counts), 'largest_group': int(counts.max()),
                       'singleton_groups': int((counts == 1).sum()), 'proxy_not_complete_dependence': True})
        for kind, a, b in comps:
            fa, fb = [cache[n].set_index('sample_id').loc[ids] for n in [a, b]]
            ea, eb = fa.predicted_label.ne(fa.true_label), fb.predicted_label.ne(fb.true_label)
            errors.append({'cohort': cohort, 'kind': kind, 'A': a, 'B': b,
                           'fix': int((~ea & eb).sum()), 'break': int((ea & ~eb).sum()),
                           'both_wrong': int((ea & eb).sum()), 'different': int(fa.predicted_label.ne(fb.predicted_label).sum())})
        for si, scheme in enumerate(cfg['bootstrap_schemes']):
            gi = group_idx if scheme == 'cluster' else np.arange(len(ids))
            g = int(gi.max() + 1)
            aggregate = np.zeros((g, len(keys), 6))
            np.add.at(aggregate, gi, features)
            flat = aggregate.reshape(g, -1)
            rng = np.random.default_rng(np.random.SeedSequence([cfg['bootstrap_seed'], ci, si]))
            distribution = np.empty((cfg['bootstrap_replicates'], len(keys), 3))
            for begin in range(0, cfg['bootstrap_replicates'], 100):
                batch = min(100, cfg['bootstrap_replicates'] - begin)
                w = rng.multinomial(g, np.full(g, 1/g), size=batch)
                with threadpool_limits(limits=2):
                    sums = (w @ flat).reshape(batch, len(keys), 6)
                distribution[begin:begin+batch] = metric_draws(sums)
                if time.perf_counter() - start > cfg['resource_budget']['max_seconds_per_stage']:
                    raise TimeoutError('Bootstrap vượt ngân sách; giữ attempt và dừng.')
            draws[cohort + '_' + scheme] = distribution
            for kind, a, b in comps:
                delta = distribution[:, index[a]] - distribution[:, index[b]]
                for mi, metric in enumerate(cfg['bootstrap_metrics']):
                    lo, hi = np.quantile(delta[:, mi], [.025, .975])
                    rows.append({'cohort': cohort, 'scheme': scheme, 'kind': kind, 'A': a, 'B': b,
                                 'metric': metric, 'delta': float(point[index[a], mi]-point[index[b], mi]),
                                 'ci_low': float(lo), 'ci_high': float(hi), 'bootstrap_replicates': cfg['bootstrap_replicates']})
            print('COMPLETE', cohort, scheme, 'B=', cfg['bootstrap_replicates'], flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == 4 * 2 * 52 * 3 and np.isfinite(result[['delta', 'ci_low', 'ci_high']]).all().all()
    write_csv(folder / 'paired_ci.csv', result)
    write_csv(folder / 'paired_errors.csv', pd.DataFrame(errors))
    write_csv(folder / 'bootstrap_groups.csv', pd.DataFrame(groups))
    np.savez_compressed(folder / 'fixed_prediction_draws.npz', **draws)
    guard()
    commit('01_bootstrap', folder, time.perf_counter() - start)
    return result


def fit_combiner(p, y, pairs, cfg):
    k = p.shape[1]
    initial = np.array([1/k]*k + ([0.]*(k*(k-1)//2) if pairs else []))
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter('always')
        result = minimize(lambda w: log_loss(y, pooling(p, w, pairs), labels=[0, 1]), initial,
                          method=cfg['optimizer']['method'], bounds=[(0., 1.)]*len(initial),
                          constraints={'type': 'eq', 'fun': lambda w: w.sum()-1},
                          options={'maxiter': cfg['optimizer']['maxiter'], 'ftol': cfg['optimizer']['ftol']})
    seconds = time.perf_counter()-start
    assert result.success and np.isfinite(result.x).all() and result.x.min() >= -1e-8, str(result.message)
    assert abs(result.x.sum()-1) < 1e-7 and seconds <= cfg['resource_budget']['max_fit_seconds']
    w = np.maximum(result.x, 0)
    return w/w.sum(), seconds, [str(item.message) for item in caught], int(result.nit)


def stability():
    """18 fixed combiner fits, M-internal group subsampling; no H or Val evaluation."""
    _, cfg = guard()
    done = completed('02_stability')
    if done is not None:
        return pd.read_csv(done / 'metrics.csv')
    start = time.perf_counter()
    folder = attempt('02_stability')
    frames = {n: pd.read_csv(old_stage(OLD, '02_calibration_' + n) / 'calibrated_M_predictions.csv')
              for n in MEMBERS['B2']}
    ref, _ = align(frames, MEMBERS['B2'])
    assignment = pd.read_csv(old_stage(OLD, '00_prepare') / 'assignment.csv').set_index('sample_id').loc[ref.sample_id]
    assert assignment.role.eq('M').all() and np.array_equal(assignment.label, ref.true_label)
    y = ref.true_label.to_numpy()
    all_rows, all_weights, split_rows, fitted = [], [], [], {}
    for seed in cfg['stability_seeds']:
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed)
        train_idx, audit_idx = next(splitter.split(np.arange(len(ref)), y, groups=assignment.group_id))
        assert not set(assignment.group_id.iloc[train_idx]) & set(assignment.group_id.iloc[audit_idx])
        assert len(train_idx)+len(audit_idx) == 642 and set(y[train_idx]) == set(CLASSES) == set(y[audit_idx])
        for role, idx in [('M_internal_fit', train_idx), ('M_internal_audit', audit_idx)]:
            for i in idx:
                split_rows.append({'seed': seed, 'sample_id': ref.sample_id.iloc[i],
                                   'group_id': int(assignment.group_id.iloc[i]), 'role': role,
                                   'label': ref.true_label.iloc[i]})
        for config, members in MEMBERS.items():
            current_ref, p = align(frames, members)
            assert np.array_equal(current_ref.sample_id, ref.sample_id)
            for method, pairs in [('weighted_mean', False), ('learned_choquet', True)]:
                w, seconds, messages, nit = fit_combiner(p[train_idx], (y[train_idx] == 'real').astype(int), pairs, cfg)
                tag = str(seed) + '__' + config + '__' + method
                fitted[tag] = {'seed': seed, 'config': config, 'method': method, 'weights': w.tolist(),
                               'fit_seconds': seconds, 'warnings': messages, 'iterations': nit}
                k = len(members)
                terms = [(i,) for i in range(k)] + (list(combinations(range(k), 2)) if pairs else [])
                for term, mass in zip(terms, w):
                    all_weights.append({'seed': seed, 'config': config, 'method': method,
                                        'term': '+'.join(members[i] for i in term), 'mass': float(mass)})
                for role, idx in [('M_internal_fit', train_idx), ('M_internal_audit', audit_idx)]:
                    f = frame(ref.iloc[idx].reset_index(drop=True), pooling(p[idx], w, pairs))
                    score, _ = pilot.evaluate(f)
                    score.update(seed=seed, config=config, method=method, role=role, fit_seconds=seconds)
                    all_rows.append(score)
                    write_csv(folder / (tag + '__' + role + '_predictions.csv'), f)
        print('COMPLETE M-internal subsampling seed', seed, 'fit/audit=', len(train_idx), len(audit_idx), flush=True)
        if time.perf_counter()-start > cfg['resource_budget']['max_seconds_per_stage']:
            raise TimeoutError('Stability quá ngân sách; dừng, không mở rộng.')
    assert len(fitted) == cfg['new_fit_budget'] == 18
    write_json(folder / 'combiner_parameters.json', fitted)
    write_csv(folder / 'assignments_M_internal.csv', pd.DataFrame(split_rows))
    write_csv(folder / 'weights.csv', pd.DataFrame(all_weights))
    metrics = pd.DataFrame(all_rows)
    write_csv(folder / 'metrics.csv', metrics)
    agg = metrics.groupby(['config', 'method', 'role'])[['weighted_f1', 'log_loss', 'brier_real_binary']].agg(['mean', 'std', 'min', 'max'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    write_csv(folder / 'subsample_summary_NOT_independent_runs.csv', agg.reset_index())
    # These repeats share most M rows and the same A/C fitted base pipelines.
    write_json(folder / 'limitations.json', {'not_base_training_multiseed': True,
               'not_independent_replicates': True, 'H_Validation_used': False,
               'no_hyperparameter_selection': True, 'parameter_variation_not_sampling_CI': True})
    guard()
    commit('02_stability', folder, time.perf_counter()-start)
    return metrics


def ablation():
    """Fixed parameter projection; no new fit and no invalid unnormalized log loss."""
    guard()
    done = completed('03_ablation')
    if done is not None:
        return pd.read_csv(done / 'metrics.csv')
    start = time.perf_counter()
    folder = attempt('03_ablation')
    parameters = read_json(old_stage(B_RUN, '01_meta_lock') / 'parameters.json')
    rows, changes, bins = [], [], []
    for cohort, cache in cached_cohorts().items():
        for config, members in MEMBERS.items():
            base = {n: cache['BASE__' + n] for n in members}
            ref, p = align(base, members)
            k = len(members)
            w = np.array(parameters[config]['learned_choquet']['weights'])
            full = pooling(p, w, True)
            s = pooling(p, w, True, return_scores=True)
            assert np.array_equal(s.argmax(axis=1), full.argmax(axis=1)), 'Normalization không được đổi argmax.'
            # Parameter removal, not relearning. Valid probabilities from additive singleton projection.
            singles = w[:k].copy()
            fallback = singles.sum() <= 1e-15
            singles = np.full(k, 1/k) if fallback else singles/singles.sum()
            for method, q in [('frozen_choquet', full), ('drop_pairs_singleton_projection', pooling(p, singles))]:
                f = frame(ref, q)
                score, b = pilot.evaluate(f)
                score.update(cohort=cohort, config=config, method=method, fit_seconds=0.,
                             projection_equal_fallback=fallback, status='exploratory_frozen_ablation')
                rows.append(score)
                bins.append(b.assign(cohort=cohort, config=config, method=method))
                write_csv(folder / (cohort + '__' + config + '__' + method + '_predictions.csv'), f)
            changes.append({'cohort': cohort, 'config': config, 'normalization_changes_hard_labels': 0,
                            'unnormalized_probability_metrics': 'N/A: scores do not sum to one',
                            'singletons_projection': singles.tolist()})
    result = pd.DataFrame(rows)
    write_csv(folder / 'metrics.csv', result)
    write_csv(folder / 'calibration_bins.csv', pd.concat(bins, ignore_index=True))
    write_json(folder / 'normalization_checks.json', changes)
    guard()
    commit('03_ablation', folder, time.perf_counter()-start)
    return result


def evidence_figures():
    guard()
    done = completed('04_figures')
    if done is not None:
        return done
    import matplotlib.pyplot as plt
    start = time.perf_counter()
    folder = attempt('04_figures')
    ci = pd.read_csv(completed('01_bootstrap') / 'paired_ci.csv')
    targets = [('B1__weighted_mean', 'B0__weighted_mean'), ('B1__weighted_mean', 'BASE__char_lr_raw'),
               ('B1__learned_choquet', 'B1__weighted_mean'), ('B2__learned_choquet', 'B2__weighted_mean')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ri, cohort in enumerate(['H_exploratory', 'benchmark_original']):
        for mi, metric in enumerate(['weighted_f1', 'log_loss', 'brier']):
            ax = axes[ri, mi]
            for j, (a, b) in enumerate(targets):
                row = ci[(ci.cohort == cohort) & (ci.scheme == 'cluster') & (ci.A == a) & (ci.B == b) & (ci.metric == metric)].iloc[0]
                scale = 100 if metric == 'weighted_f1' else 1
                ax.plot([row.ci_low*scale, row.ci_high*scale], [j, j], color='tab:blue')
                ax.scatter(row.delta*scale, j, color='tab:blue')
            ax.axvline(0, color='grey', linestyle='--')
            ax.set(yticks=range(len(targets)), yticklabels=[a.replace('__', '/')+' − '+b.replace('__', '/') for a, b in targets],
                   xlabel='ΔWF1 (điểm %)' if mi == 0 else 'Δ'+metric, title=cohort+' / '+metric)
            ax.tick_params(axis='y', labelsize=7)
    fig.suptitle('Paired cluster bootstrap 95% — exploratory; prediction cố định, chưa chỉnh đa so sánh')
    fig.tight_layout()
    fig.savefig(folder / '01_paired_intervals.png', dpi=150)
    plt.close(fig)
    weights = pd.read_csv(completed('02_stability') / 'weights.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, config in zip(axes, MEMBERS):
        t = weights[(weights.config == config) & (weights.method == 'learned_choquet')].pivot(index='seed', columns='term', values='mass')
        for seed, row in t.iterrows():
            ax.plot(range(len(t.columns)), row, 'o-', label=str(seed))
        ax.set(xticks=range(len(t.columns)), xticklabels=t.columns, ylabel='Möbius mass', title=config)
        ax.tick_params(axis='x', rotation=40, labelsize=8)
        ax.legend(title='M subset seed')
    fig.suptitle('Độ nhạy capacity: cùng A/C, ba phần M nội bộ; không phải ba lần train base độc lập')
    fig.tight_layout()
    fig.savefig(folder / '02_capacity_subsampling.png', dpi=150)
    plt.close(fig)
    ab = pd.read_csv(completed('03_ablation') / 'metrics.csv')
    for cohort, table in ab.groupby('cohort', sort=False):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for ax, row in zip(axes.flat, table.itertuples()):
            cm = np.array([[row.cm_fake_fake, row.cm_fake_real], [row.cm_real_fake, row.cm_real_real]], dtype=int)
            ax.imshow(cm, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center')
            ax.set(title=row.config+' / '+row.method, xticks=[0, 1], xticklabels=CLASSES,
                   yticks=[0, 1], yticklabels=CLASSES, xlabel='Dự đoán', ylabel='Nhãn thật')
            ax.title.set_fontsize(9)
        fig.suptitle(cohort+' — ablation không fit; mọi phương pháp mới báo cáo')
        fig.tight_layout()
        fig.savefig(folder / ('03_ablation_cm_'+cohort+'.png'), dpi=140)
        plt.close(fig)
    for relative in read_json(completed('00_preflight') / 'protocol_lock.json')['input_hashes']:
        assert safe(relative).exists()
    commit('04_figures', folder, time.perf_counter()-start)
    return folder


def table_md(frame):
    headers = [str(c) for c in frame.columns]
    rows = ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    for values in frame.itertuples(index=False, name=None):
        rows.append('| '+' | '.join(str(v).replace('|', '/') for v in values)+' |')
    return '\n'.join(rows)


def dossier():
    """Package all development evidence; no new method fit without novelty gate."""
    _, cfg = guard()
    done = completed('05_dossier')
    if done is not None:
        return done
    required = ['01_bootstrap', '02_stability', '03_ablation', '04_figures']
    assert all(completed(name) is not None for name in required)
    start = time.perf_counter()
    folder = attempt('05_dossier')
    literature = ROOT / 'reports/20261019_development_literature_gap_v1.md'
    assert literature.exists(), 'Chờ literature review hoàn tất trước đóng hồ sơ.'
    # Do not turn an already published regularizer into an invented new method.
    write_json(folder / 'method_gate.json', {
        'candidate': 'regularize learned interaction toward additive pooling',
        'decision': 'NOT_IMPLEMENTED_AS_NEW_PROPOSED_METHOD',
        'reason': 'Capacity regularization toward predefined structures and cross-entropy measure learning have direct prior work; no supported differentiator or new independent evaluation established.',
        'literature_review': str(literature.relative_to(ROOT)), 'literature_sha256': sha(literature),
        'new_encoder': 'Not necessary to answer operator/member/calibration hypotheses; not added merely to enlarge ranking',
        'method_research_complete': False, 'empirical_development_audit_complete': True})
    write_json(folder / 'final_evaluation_protocol_DRAFT_REQUIRES_APPROVAL.json', {
        'status': 'DRAFT_REQUIRES_SEPARATE_EXPLICIT_USER_APPROVAL', 'test_access_now': False,
        'selected_models': None, 'selected_winner': None,
        'before_access': ['Human chooses models and primary contrasts from development, freezes exact checkpoint/calibrator/combiner/threshold/label mapping and hashes',
                          'Decide keep A/C/M-fitted pipelines vs a separately designed final refit; never refit baselines alone while comparing to frozen ensembles as equal protocol',
                          'Confirm dataset rights and exact final Train protocol, metrics/CI/reporting budget',
                          'Commit immutable protocol and obtain separate Test approval'],
        'after_separate_approval_only': ['One frozen evaluation pass, no tuning based on Test',
                                       'Report every predeclared model/contrast and limitations; no hidden retry/selection']})
    metrics_b = pd.concat([pd.read_csv(old_stage(B_RUN, n) / 'metrics.csv') for n in ['02_H', '03_Validation']], ignore_index=True)
    metrics_pilot = pd.read_csv(old_stage(OLD, '07_summary') / 'pilot_all_metrics.csv')
    write_csv(folder / 'B0_B1_B2_all_metrics.csv', metrics_b)
    write_csv(folder / 'pilot_A_fit_all_metrics.csv', metrics_pilot)
    historical = read_json(ROOT / 'results/20261018_190510_preapproval_f939ffb0/metrics_all.json')
    hist_rows = []
    for model, cohorts in historical.items():
        for cohort, m in cohorts.items():
            row = {'model': model, 'cohort': cohort, 'protocol': 'historical_Train_full_NOT_pilot_A_fit'}
            for key in ['n', 'weighted_f1', 'macro_f1', 'accuracy', 'balanced_accuracy', 'mcc', 'log_loss']:
                row[key] = m.get(key)
            for label in ['fake', 'real']:
                row['f1_'+label] = m['per_class'][label]['f1-score']
                row['precision_'+label] = m['per_class'][label]['precision']
                row['recall_'+label] = m['per_class'][label]['recall']
            for average in ['weighted', 'macro']:
                row['precision_'+average] = m['per_class'][average+' avg']['precision']
                row['recall_'+average] = m['per_class'][average+' avg']['recall']
            hist_rows.append(row)
    write_csv(folder / 'historical_Train_full_all_metrics.csv', pd.DataFrame(hist_rows))
    # Genuine existing in-sample diagnostics, never used to select models here.
    a_rows = []
    for model in ['dummy', 'word_lr', 'word_nb', 'word_svm', 'char_lr']:
        train = pd.read_csv(old_stage(OLD, '01_base_'+model) / 'train_diagnostic.csv').iloc[0]
        for cohort in ['H_meta_audit', 'benchmark_original']:
            val = metrics_pilot[(metrics_pilot.model == model) & (metrics_pilot.representation == 'raw') & (metrics_pilot.cohort == cohort)].iloc[0]
            a_rows.append({'model': model, 'cohort': cohort, 'A_in_sample_weighted_f1': train.weighted_f1,
                           'audit_weighted_f1': val.weighted_f1, 'gap': train.weighted_f1-val.weighted_f1,
                           'A_log_loss': train.log_loss, 'audit_log_loss': val.log_loss,
                           'fit_seconds_A': train.base_fit_seconds})
    write_csv(folder / 'base_A_in_sample_gaps_DIAGNOSTIC.csv', pd.DataFrame(a_rows))
    bert = pd.read_csv(ROOT / 'results/20261018_183010_bert_tiny_8945ea6b/history.csv')
    write_csv(folder / 'BERT_Tiny_historical_epoch_history.csv', bert)
    ci = pd.read_csv(completed('01_bootstrap') / 'paired_ci.csv')
    stable = pd.read_csv(completed('02_stability') / 'subsample_summary_NOT_independent_runs.csv')
    ablation_metrics = pd.read_csv(completed('03_ablation') / 'metrics.csv')
    sections = ['# Hồ sơ hoàn tất DEVELOPMENT — CONSTRAINT 2021',
                'Official Test khóa; chưa chọn winner. Hoàn tất audit development khả thi, không tuyên bố hoàn tất một phương pháp mới.',
                '## Tái lập', table_md(pd.read_csv(completed('00_preflight') / 'split_summary.csv')),
                '## B0/B1/B2 — đầy đủ metric', table_md(metrics_b.round(8)),
                '## Pilot A-fit baseline raw/calibrated', table_md(metrics_pilot.round(8)),
                '## Train-full lịch sử — bảng riêng', table_md(pd.DataFrame(hist_rows).round(8)),
                '## Paired CI95% cluster chính — chưa chỉnh đa so sánh',
                table_md(ci[(ci.scheme == 'cluster') & (ci.cohort.isin(['H_exploratory', 'benchmark_original']))].round(8)),
                '## Độ nhạy M subsampling — không độc lập', table_md(stable.round(8)),
                '## Ablation frozen interaction', table_md(ablation_metrics.round(8)),
                '## Research gap', literature.read_text(encoding='utf-8'),
                '## Giới hạn',
                'H/Validation đã xem, mọi kết quả exploratory. CI conditional on fitted predictions; lexical groups/proxy chưa bao quát phụ thuộc. '
                'Ba cohort lồng nhau. Ba phần M chồng lấp không là seed uncertainty của full training. '
                'Không áp score M fit như independent. Không normalize scores rồi gọi raw/calibration như nhau. '
                'Brier/NLL không đo calibration thuần. Không so A-fit với full-Train như ngang điều kiện. '
                'B2 projection chỉ cô lập loại pair ở tham số cố định, không là đối chứng thay thế Weighted đã fit độc lập.',
                '## Hai mục tiêu',
                'Đủ hồ sơ thực nghiệm môn học: data/leakage, classical+TinyBERT lịch sử, calibration/ensemble/fuzzy, CI và ablation có giới hạn. '
                'Bài báo Q3/Q4: chưa có novelty được chứng minh, chưa independent final evaluation, chưa đa dataset/domain; quartile không phải bảo đảm chấp nhận.',
                '## Dừng', 'Chờ người nghiên cứu quyết định mô hình/giao thức đánh giá cuối. Test vẫn khóa.']
    with (folder / 'development_report.md').open('x', encoding='utf-8') as f:
        f.write('\n\n'.join(sections))
    guard()
    write_json(folder / 'state.json', {'status': 'FEASIBLE_DEVELOPMENT_COMPLETE_AWAITING_HUMAN_DECISION',
               'new_base_fits': 0, 'combiner_diagnostic_fits': 18, 'new_encoder_fits': 0,
               'bootstrap_rows': len(ci), 'new_method_proven': False, 'new_method_branch_stopped': True,
               'official_test_read': False, 'no_09_to_08_rerun': True})
    commit('05_dossier', folder, time.perf_counter()-start)
    return folder
