"""User-executed notebook stages for H1/H2. Import has no data I/O or fitting."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import platform
import re
import time
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score, log_loss, matthews_corrcoef
from threadpoolctl import threadpool_limits
from text_preprocessing import normalize_text
from validation_cohorts import select_validation_cohorts

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw/constraint2021/030deba36d118c124709fd72d7d5f14c7bc69053'
REVIEW = ROOT / 'results/20261018_181449_leakage_11b82aa3/review_v2'
CONFIG = ROOT / 'configs/pilot_h1_h2_seed42_v1.json'
CLASSES = np.array(['fake', 'real'])
VAL_SHA = 'c610931d0641fe1d9b9495c453f99ae7aa45bf294c7db6333ec84ecebb0b6c3c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, frame):
    frame.to_csv(path, index=False, mode='x')


def _run_path(run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
        raise ValueError('RUN_ID chỉ gồm chữ, số, gạch nối/gạch dưới; không phải đường dẫn.')
    return ROOT / 'results' / run_id


def _check_files(files):
    for relative, expected in files.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Artifact/source đã đổi: {relative}; dừng, không tự ghi đè.')


def _file_map(paths):
    return {str(Path(p).relative_to(ROOT)): sha(p) for p in paths}


def _commit(run, name, folder):
    files = _file_map(p for p in folder.rglob('*') if p.is_file())
    marker = {'stage': name, 'folder': str(folder.relative_to(ROOT)), 'files': files,
              'time': datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat()}
    write_json(run / f'{name}_COMPLETE.json', marker)
    with (run / 'run.log').open('a', encoding='utf-8') as log:
        log.write(f"{marker['time']} {name} COMPLETE\n")
    return folder


def _completed(run, name):
    path = run / f'{name}_COMPLETE.json'
    if not path.exists():
        return None
    marker = json.loads(path.read_text(encoding='utf-8'))
    _check_files(marker['files'])
    return ROOT / marker['folder']


def _attempt(run, name):
    folder = run / f'{name}_attempt_{uuid.uuid4().hex[:8]}'
    folder.mkdir(exist_ok=False)
    return folder


def session(run_id):
    run = _run_path(run_id)
    prepared = _completed(run, '00_prepare')
    if prepared is None:
        raise ValueError('Chạy Notebook 04 trước; không tự đoán run gần nhất.')
    frozen = json.loads((prepared / 'session.json').read_text(encoding='utf-8'))
    _check_files(frozen['source_hashes'])
    config = json.loads((prepared / 'config.json').read_text(encoding='utf-8'))
    return run, prepared, config


def prepare(run_id):
    """Create assignment only when the user runs Notebook 04; no model fitting."""
    run = _run_path(run_id)
    if _completed(run, '00_prepare') is not None:
        session(run_id)
        print('Đã có assignment hoàn tất; xác minh hash và tái sử dụng, không chia lại.')
        return run_id
    run.mkdir(exist_ok=False)
    out = _attempt(run, '00_prepare')
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    assert config['seed'] == 42 and config['ratios'] == {'A': .7, 'C': .1, 'M': .1, 'H': .1}
    assert not config['multi_seed'] and not config['test_access'] and config['new_cv_folds'] == 0
    provenance = json.loads((RAW / 'provenance.json').read_text(encoding='utf-8'))
    expected = next(f['sha256'] for f in provenance['files'] if f['name'] == 'Constraint_Train.csv')
    assert expected == '441f3985bb76eb69d1e488cf292f80add96c4dddfa43ba7c060889607837ddce'
    assert sha(RAW / 'Constraint_Train.csv') == expected
    train = pd.read_csv(RAW / 'Constraint_Train.csv', dtype=str)
    assert len(train) == 6420 and not train.isna().any().any()
    train['sample_id'] = 'train:' + train.id
    assert train.sample_id.is_unique and set(train.label) == set(CLASSES)
    groups = pd.read_csv(ROOT / config['group_source'], dtype={'sample_id': str})
    assert groups.sample_id.is_unique and set(groups.sample_id) == set(train.sample_id)
    groups = groups.set_index('sample_id').loc[train.sample_id]
    assert np.array_equal(groups.true_label, train.label)
    train['group_id'] = groups.group_id.to_numpy()
    # Only label counts used for stratification; no scores/models/H-based choices.
    counts = pd.crosstab(train.group_id, train.label).reindex(columns=CLASSES, fill_value=0)
    rng = np.random.default_rng(config['seed'])
    counts = counts.iloc[rng.permutation(len(counts))]
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False, kind='stable').index]
    roles = list(config['ratios'])
    target = np.array(list(config['ratios'].values()))[:, None] * counts.sum().to_numpy()[None, :]
    current = np.zeros_like(target)
    assignment = {}
    for group, row in counts.iterrows():
        costs = []
        for k in range(4):
            trial = current.copy()
            trial[k] += row.to_numpy()
            costs.append(float(np.sum((trial - target) ** 2 / target)))
        chosen = int(np.argmin(costs))
        current[chosen] += row.to_numpy()
        assignment[group] = roles[chosen]
    train['role'] = train.group_id.map(assignment)
    assert train.groupby('group_id').role.nunique().max() == 1
    assert set(train.role) == set(roles)
    for k, role in enumerate(roles):
        assert (current[k] > 0).all() and np.max(np.abs(current[k] - target[k]) / counts.sum().to_numpy()) < .02
    edges = pd.read_csv(ROOT / config['group_edges'])
    for a, b in edges.iloc[:, :2].itertuples(index=False, name=None):
        assert train.role.iloc[int(a)] == train.role.iloc[int(b)]
    write_csv(out / 'assignment.csv', train[['sample_id', 'label', 'group_id', 'role']])
    summary = pd.crosstab(train.role, train.label).reindex(roles).reset_index()
    summary['n'] = summary.fake + summary.real
    summary['fraction'] = summary.n / len(train)
    write_csv(out / 'split_summary.csv', summary)
    for role in roles:
        write_csv(out / f'partition_{role}.csv', train.loc[train.role == role, ['sample_id', 'tweet', 'label', 'group_id']])
    write_json(out / 'config.json', config)
    sources = [CONFIG, Path(__file__), ROOT / 'src/text_preprocessing.py', ROOT / 'src/validation_cohorts.py',
               ROOT / config['group_source'], ROOT / config['group_edges'], RAW / 'Constraint_Train.csv',
               RAW / 'provenance.json', REVIEW / 'validation_evaluation_masks.csv', REVIEW / 'protocol_proposal.json']
    write_json(out / 'session.json', {'run_id': run_id, 'source_hashes': _file_map(sources),
        'n_train': len(train), 'group_independence': True, 'new_folds': 0, 'test_read': False,
        'limitation': 'H belongs to historical Train, already used in older full-Train runs; only independent of this new pipeline fitting. Lexical grouping may miss paraphrases.'})
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'matplotlib', 'threadpoolctl']
    write_json(out / 'environment.json', {'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {p: importlib.metadata.version(p) for p in packages}})
    _commit(run, '00_prepare', out)
    print(summary.to_string(index=False))
    print('PASS: assignment group-aware, stratification xấp xỉ; H chưa được đánh giá.')
    return run_id


def split_summary(run_id):
    _, prepared, _ = session(run_id)
    return pd.read_csv(prepared / 'split_summary.csv')


def _partition(prepared, role):
    assert role in ['A', 'C', 'M', 'H']
    frame = pd.read_csv(prepared / f'partition_{role}.csv')
    assert frame.sample_id.is_unique and set(frame.label) == set(CLASSES)
    return frame


def _base_pipeline(name, cfg):
    params = cfg['base_parameters']
    features = cfg['features']
    estimators = {'dummy': DummyClassifier(strategy='most_frequent'),
        'word_lr': LogisticRegression(C=params['LR_C'], max_iter=params['LR_max_iter'], random_state=42),
        'word_nb': MultinomialNB(alpha=params['NB_alpha']),
        'word_svm': LinearSVC(C=params['SVM_C'], dual='auto', max_iter=params['SVM_max_iter'], random_state=42),
        'char_lr': LogisticRegression(C=params['LR_C'], max_iter=params['LR_max_iter'], random_state=42)}
    char = name == 'char_lr'
    vectorizer = TfidfVectorizer(preprocessor=normalize_text, lowercase=False,
        analyzer='char' if char else 'word', ngram_range=tuple(features['char_ngrams' if char else 'word_ngrams']),
        min_df=features['min_df'], max_features=features['max_features'], sublinear_tf=features['sublinear_tf'], dtype=np.float32)
    return Pipeline([('tfidf', vectorizer), ('classifier', estimators[name])])


def _prob_and_score(model, texts):
    assert list(model.classes_) == ['fake', 'real']
    if hasattr(model, 'predict_proba'):
        probability = np.asarray(model.predict_proba(texts), dtype=float)
        probability /= probability.sum(axis=1, keepdims=True)
        real = np.clip(probability[:, 1], 1e-7, 1 - 1e-7)
        score = np.log(real / (1 - real))
    else:
        probability = None
        score = np.asarray(model.decision_function(texts), dtype=float)
    assert np.isfinite(score).all()
    return probability, score.reshape(-1, 1)


def _frame(data, predicted, probability=None):
    frame = data[['sample_id', 'label']].rename(columns={'label': 'true_label'}).copy()
    frame['predicted_label'] = predicted
    if probability is not None:
        assert probability.shape == (len(data), 2) and np.isfinite(probability).all()
        assert (probability >= 0).all() and (probability <= 1).all() and np.allclose(probability.sum(axis=1), 1)
        frame['prob_fake'], frame['prob_real'] = probability[:, 0], probability[:, 1]
    return frame


def fit_base(run_id, name):
    run, prepared, config = session(run_id)
    assert name in config['base_models']
    stage = '01_base_' + name
    done = _completed(run, stage)
    if done is not None:
        print(stage, 'đã hoàn tất; không fit lại.')
        return pd.read_csv(done / 'train_diagnostic.csv')
    out = _attempt(run, stage)
    a = _partition(prepared, 'A')
    model = _base_pipeline(name, config)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=config['native_threads']):
        warnings.simplefilter('always', ConvergenceWarning)
        model.fit(a.tweet, a.label)
    fit_seconds = time.perf_counter() - start
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        write_json(out / 'warnings.json', [str(w.message) for w in caught])
        raise RuntimeError('Chưa hội tụ; giữ attempt, dừng trước meta/H; không tự tuning.')
    assert list(model.classes_) == ['fake', 'real']
    joblib.dump(model, out / 'pipeline.joblib')
    probability, _ = _prob_and_score(model, a.tweet)
    frame = _frame(a, model.predict(a.tweet), probability)
    scores, _ = evaluate(frame)
    scores.update({'model': name, 'cohort': 'A_in_sample_diagnostic', 'base_fit_seconds': fit_seconds,
                   'epoch_history': 'N/A: estimator không lưu epoch history'})
    write_csv(out / 'train_diagnostic.csv', pd.DataFrame([scores]))
    _commit(run, stage, out)
    print(name, 'fit A:', round(fit_seconds, 3), 'giây; không đọc C/M/H/Validation.')
    return pd.DataFrame([scores])


def fit_calibration(run_id, name):
    run, prepared, config = session(run_id)
    assert name in config['base_models']
    stage = '02_calibration_' + name
    done = _completed(run, stage)
    if done is not None:
        print(stage, 'đã hoàn tất; không fit lại.')
        return pd.read_csv(done / 'metrics_M_FIT_DIAGNOSTIC.csv')
    base = _completed(run, '01_base_' + name)
    if base is None:
        raise ValueError('Fit base A trước.')
    out = _attempt(run, stage)
    model = joblib.load(base / 'pipeline.joblib')
    c, m = _partition(prepared, 'C'), _partition(prepared, 'M')
    raw_c, scores_c = _prob_and_score(model, c.tweet)
    c_inputs = c[['sample_id', 'label']].copy()
    c_inputs['calibration_input_score'] = scores_c[:, 0]
    if raw_c is not None:
        c_inputs['raw_prob_real'] = raw_c[:, 1]
    write_csv(out / 'calibration_C_inputs.csv', c_inputs)
    cp = config['calibration']
    calibrator = LogisticRegression(C=cp['C'], max_iter=cp['max_iter'], random_state=42)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=config['native_threads']):
        warnings.simplefilter('always', ConvergenceWarning)
        calibrator.fit(scores_c, c.label)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError('Calibrator chưa hội tụ; dừng trước H, không đổi cấu hình tự động.')
    seconds = time.perf_counter() - start
    assert list(calibrator.classes_) == ['fake', 'real']
    joblib.dump(calibrator, out / 'calibrator.joblib')
    raw, scores_m = _prob_and_score(model, m.tweet)
    calibrated = calibrator.predict_proba(scores_m)
    rows = []
    for kind, probability in [('raw', raw), ('calibrated', calibrated)]:
        prediction = model.predict(m.tweet) if probability is None else CLASSES[probability.argmax(axis=1)]
        frame = _frame(m, prediction, probability)
        if kind == 'raw' and probability is None:
            frame['decision_score_real'] = scores_m[:, 0]
        write_csv(out / f'{kind}_M_predictions.csv', frame)
        summary, bins = evaluate(frame)
        summary.update({'model': name, 'representation': kind, 'cohort': 'M_meta_fit_diagnostic',
                        'calibration_fit_seconds_C': seconds if kind == 'calibrated' else 0.0})
        rows.append(summary)
        write_csv(out / f'{kind}_M_bins.csv', bins)
    write_csv(out / 'metrics_M_FIT_DIAGNOSTIC.csv', pd.DataFrame(rows))
    write_json(out / 'calibration.json', {'method': cp, 'fit': 'C only', 'fit_seconds': seconds,
        'coefficient': calibrator.coef_.tolist(), 'intercept': calibrator.intercept_.tolist(),
        'negative_slope': bool(calibrator.coef_[0, 0] < 0), 'note': 'negative slope is reported, not silently changed; M is diagnostic only'})
    _commit(run, stage, out)
    return pd.DataFrame(rows)


def _bins(probability, target, mode):
    real = probability[:, 1]
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1) if mode == 'top_label' else real
    observed = (prediction == target).astype(float) if mode == 'top_label' else target.astype(float)
    index = np.minimum((confidence * 10).astype(int), 9)
    rows = []
    for k in range(10):
        select = index == k
        n = int(select.sum())
        mean_p = float(confidence[select].mean()) if n else None
        mean_y = float(observed[select].mean()) if n else None
        rows.append({'mode': mode, 'bin': k, 'lower': k / 10, 'upper': (k + 1) / 10, 'n': n,
            'mean_probability': mean_p, 'observed_frequency': mean_y,
            'gap': abs(mean_p - mean_y) if n else None})
    return pd.DataFrame(rows)


def evaluate(frame):
    y, pred = frame.true_label, frame.predicted_label
    report = classification_report(y, pred, labels=CLASSES.tolist(), output_dict=True, zero_division=0)
    metrics = {'n': len(frame), 'weighted_f1': f1_score(y, pred, average='weighted', zero_division=0),
        'macro_f1': f1_score(y, pred, average='macro', zero_division=0),
        'accuracy': accuracy_score(y, pred), 'balanced_accuracy': balanced_accuracy_score(y, pred),
        'mcc': matthews_corrcoef(y, pred)}
    for label in CLASSES:
        for source, key in [('precision', 'precision'), ('recall', 'recall'), ('f1-score', 'f1'), ('support', 'support')]:
            metrics[f'{key}_{label}'] = report[label][source]
    for average in ['weighted', 'macro']:
        for key in ['precision', 'recall']:
            metrics[f'{key}_{average}'] = report[average + ' avg'][key]
    cm = confusion_matrix(y, pred, labels=CLASSES.tolist())
    for i in range(2):
        for j in range(2):
            metrics[f'cm_{CLASSES[i]}_{CLASSES[j]}'] = int(cm[i, j])
    probability_fields = ['log_loss', 'brier_real_binary', 'ece_top_10', 'mce_top_10', 'ece_real_10']
    metrics.update({key: None for key in probability_fields})
    if 'prob_real' not in frame:
        return metrics, pd.DataFrame(columns=['mode', 'bin', 'lower', 'upper', 'n', 'mean_probability', 'observed_frequency', 'gap'])
    probability = frame[['prob_fake', 'prob_real']].to_numpy(dtype=float)
    assert np.isfinite(probability).all() and (probability >= 0).all() and np.allclose(probability.sum(axis=1), 1)
    target = (y == 'real').to_numpy(dtype=int)
    metrics['log_loss'] = float(log_loss(target, probability, labels=[0, 1]))
    metrics['brier_real_binary'] = float(np.mean((probability[:, 1] - target) ** 2))
    top, real = _bins(probability, target, 'top_label'), _bins(probability, target, 'real_class')
    metrics['ece_top_10'] = float((top.n * top.gap.fillna(0)).sum() / len(frame))
    metrics['mce_top_10'] = float(top.gap.max())
    metrics['ece_real_10'] = float((real.n * real.gap.fillna(0)).sum() / len(frame))
    return metrics, pd.concat([top, real], ignore_index=True)


def pool(probability, weights):
    assert probability.ndim == 3 and probability.shape[1:] == (2, 2)
    assert np.isfinite(probability).all() and (probability >= 0).all() and np.allclose(probability.sum(axis=2), 1)
    w = np.asarray(weights, dtype=float)
    assert w.shape in [(2,), (3,)] and (w >= -1e-12).all() and np.isclose(w.sum(), 1)
    basis = [probability[:, 0, :], probability[:, 1, :]]
    if len(w) == 3:
        basis.append(np.minimum(probability[:, 0, :], probability[:, 1, :]))
    scores = np.einsum('nkc,k->nc', np.stack(basis, axis=1), w)
    total = scores.sum(axis=1, keepdims=True)
    return np.divide(scores, total, out=np.full_like(scores, .5), where=total > 1e-15)


def _aligned_pair(frames, members):
    a, b = [frames[n] for n in members]
    assert a.sample_id.is_unique and b.sample_id.is_unique and set(a.sample_id) == set(b.sample_id)
    b = b.set_index('sample_id').loc[a.sample_id].reset_index()
    assert np.array_equal(a.true_label, b.true_label)
    p = np.stack([x[['prob_fake', 'prob_real']].to_numpy() for x in [a, b]], axis=1)
    return a, p


def fit_meta(run_id):
    run, _, config = session(run_id)
    done = _completed(run, '03_meta')
    if done is not None:
        print('Meta đã hoàn tất; không tối ưu lại.')
        return pd.read_csv(done / 'metrics_M_FIT_DIAGNOSTIC.csv')
    out = _attempt(run, '03_meta')
    models = {}
    for member in config['base_models']:
        models[member] = _completed(run, '02_calibration_' + member)
        if models[member] is None:
            raise ValueError('Hoàn tất toàn bộ base/calibration trước meta.')
    rows, all_parameters = [], {}
    for kind in config['representations']:
        frames = {n: pd.read_csv(models[n] / f'{kind}_M_predictions.csv') for n in config['h2_members']}
        ref, p = _aligned_pair(frames, config['h2_members'])
        target = (ref.true_label == 'real').to_numpy(dtype=int)
        parameters = {'fixed_pair_mean': {'weights': [.5, .5], 'fit_seconds_M': 0.0},
                      'normalized_pair_min': {'weights': [0., 0., 1.], 'fit_seconds_M': 0.0}}
        for name, nweights in [('weighted_mean', 2), ('learned_nonnegative_choquet', 3)]:
            initial = np.array([.5, .5] + ([0.] if nweights == 3 else []))
            cfg = config['combiner']
            start = time.perf_counter()
            result = minimize(lambda w: log_loss(target, pool(p, w), labels=[0, 1]), initial,
                method=cfg['optimizer'], bounds=[tuple(cfg['bounds'])] * nweights,
                constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                options={'maxiter': cfg['maxiter'], 'ftol': cfg['ftol']})
            seconds = time.perf_counter() - start
            if not result.success:
                write_json(out / f'{kind}_{name}_failure.json', {'message': str(result.message)})
                raise RuntimeError('Optimizer chưa thành công; dừng trước H, không tự tuning.')
            weights = np.maximum(result.x, 0)
            weights /= weights.sum()
            capacity = [0., float(weights[0]), float(weights[1]), 1.]
            assert 0 <= capacity[1] <= 1 and 0 <= capacity[2] <= 1
            parameters[name] = {'weights': weights.tolist(), 'fit_seconds_M': seconds,
                'fit_loss_M_NOT_INDEPENDENT': float(log_loss(target, pool(p, weights), labels=[0, 1])),
                'optimizer_iterations': int(result.nit), 'capacity_empty_1_2_full': capacity}
        parameters['additive_pooling_alias'] = dict(parameters['weighted_mean'], alias_of='weighted_mean')
        for name, fitted in parameters.items():
            probability = pool(p, fitted['weights'])
            frame = ref[['sample_id', 'true_label']].copy()
            frame['predicted_label'] = CLASSES[probability.argmax(axis=1)]
            frame['prob_fake'], frame['prob_real'] = probability[:, 0], probability[:, 1]
            write_csv(out / f'{kind}_{name}_M_predictions.csv', frame)
            summary, _ = evaluate(frame)
            summary.update({'model': name, 'representation': kind, 'cohort': 'M_meta_fit_NOT_independent',
                            'combiner_fit_seconds_M': fitted['fit_seconds_M']})
            rows.append(summary)
        all_parameters[kind] = parameters
    write_json(out / 'combiners.json', all_parameters)
    write_csv(out / 'metrics_M_FIT_DIAGNOSTIC.csv', pd.DataFrame(rows))
    _commit(run, '03_meta', out)
    return pd.DataFrame(rows)


def lock_pilot(run_id):
    run, _, config = session(run_id)
    done = _completed(run, '04_pilot_lock')
    if done is not None:
        print('Pilot đã khóa; giữ nguyên, không chọn winner.')
        return json.loads((done / 'lock.json').read_text(encoding='utf-8'))
    required = ['00_prepare', '03_meta'] + ['01_base_' + n for n in config['base_models']] + ['02_calibration_' + n for n in config['base_models']]
    for stage in required:
        if _completed(run, stage) is None:
            raise ValueError('Thiếu stage ' + stage)
    out = _attempt(run, '04_pilot_lock')
    # Hash every stage marker and the frozen payload, not a chosen best M/H result.
    files = [run / f'{stage}_COMPLETE.json' for stage in required]
    payload = {'status': 'PILOT_CONFIGURATIONS_LOCKED_NOT_FINAL_RESEARCH_MODEL',
        'run_id': run_id, 'required_markers': _file_map(files), 'selected_winner': None,
        'H_used_for_tuning': False, 'official_test_read': False,
        'comparison': 'All preregistered raw/calibrated pair methods; no M-based family selection or tuning grid.'}
    write_json(out / 'lock.json', payload)
    _commit(run, '04_pilot_lock', out)
    return payload


def _audit_base_predictions(run, data, config):
    frames, timing = {}, []
    for name in config['base_models']:
        base = _completed(run, '01_base_' + name)
        cal = _completed(run, '02_calibration_' + name)
        model = joblib.load(base / 'pipeline.joblib')
        calibrator = joblib.load(cal / 'calibrator.joblib')
        start = time.perf_counter()
        with threadpool_limits(limits=config['native_threads']):
            raw, score = _prob_and_score(model, data.tweet)
            raw_prediction = model.predict(data.tweet) if raw is None else CLASSES[raw.argmax(axis=1)]
        base_seconds = time.perf_counter() - start
        start = time.perf_counter()
        calibrated = calibrator.predict_proba(score)
        cal_seconds = time.perf_counter() - start
        frames[(name, 'raw')] = _frame(data, raw_prediction, raw)
        if raw is None:
            frames[(name, 'raw')]['decision_score_real'] = score[:, 0]
        frames[(name, 'calibrated')] = _frame(data, CLASSES[calibrated.argmax(axis=1)], calibrated)
        timing.append({'model': name, 'n': len(data), 'base_inference_seconds': base_seconds,
            'calibration_apply_seconds': cal_seconds, 'scope': 'CPU; checkpoint load excluded; one pass, no speed ranking; SVM raw label may require extra predict pass'})
    meta = _completed(run, '03_meta')
    parameters = json.loads((meta / 'combiners.json').read_text(encoding='utf-8'))
    for kind in config['representations']:
        ref, probability = _aligned_pair({n: frames[(n, kind)] for n in config['h2_members']}, config['h2_members'])
        for name, fitted in parameters[kind].items():
            start = time.perf_counter()
            p = pool(probability, fitted['weights'])
            seconds = time.perf_counter() - start
            frame = ref[['sample_id', 'true_label']].copy()
            frame['predicted_label'] = CLASSES[p.argmax(axis=1)]
            frame['prob_fake'], frame['prob_real'] = p[:, 0], p[:, 1]
            frames[(name, kind)] = frame
            timing.append({'model': name, 'representation': kind, 'n': len(data),
                           'aggregation_seconds': seconds, 'scope': 'Aggregation only, excludes all base/calibration inference'})
    return frames, pd.DataFrame(timing)


def audit(run_id, cohort):
    """Only this stage can load H/Official Validation, after immutable pilot lock."""
    assert cohort in ['H', 'Validation']
    run, prepared, config = session(run_id)
    locked = _completed(run, '04_pilot_lock')
    if locked is None:
        raise ValueError('Khóa cấu hình pilot ở Notebook 06 trước khi mở H/Validation.')
    _check_files(json.loads((locked / 'lock.json').read_text(encoding='utf-8'))['required_markers'])
    # Recheck payload hashes, not just marker hashes.
    for stage in ['03_meta'] + ['01_base_' + n for n in config['base_models']] + ['02_calibration_' + n for n in config['base_models']]:
        assert _completed(run, stage) is not None
    if cohort == 'Validation' and _completed(run, '05_audit_H') is None:
        raise ValueError('Audit H trước; Validation chỉ là đối chiếu development.')
    name = '05_audit_' + cohort
    done = _completed(run, name)
    if done is not None:
        print(cohort, 'đã audit; chỉ đọc kết quả, không đánh giá lại.')
        return pd.read_csv(done / 'metrics.csv')
    out = _attempt(run, name)
    start = time.perf_counter()
    if cohort == 'H':
        data = _partition(prepared, 'H')
    else:
        assert sha(RAW / 'Constraint_Val.csv') == VAL_SHA
        data = pd.read_csv(RAW / 'Constraint_Val.csv', dtype=str)
        data['sample_id'] = 'validation:' + data.id
        assert len(data) == 2140 and data.sample_id.is_unique
    frames, timing = _audit_base_predictions(run, data, config)
    rows, bin_frames = [], []
    for (model, representation), frame in frames.items():
        write_csv(out / f'{model}_{representation}_predictions.csv', frame)
        cohorts = {'H_meta_audit': frame} if cohort == 'H' else select_validation_cohorts(frame, REVIEW, approve_proposed=True)
        for subset_name, subset in cohorts.items():
            summary, bins = evaluate(subset)
            summary.update({'model': model, 'representation': representation, 'cohort': subset_name})
            rows.append(summary)
            bins = bins.assign(model=model, representation=representation, cohort=subset_name)
            bin_frames.append(bins)
    write_csv(out / 'metrics.csv', pd.DataFrame(rows))
    write_csv(out / 'calibration_bins.csv', pd.concat(bin_frames, ignore_index=True))
    write_csv(out / 'timing.csv', timing)
    write_json(out / 'scope.json', {'cohort': cohort, 'elapsed_seconds': time.perf_counter() - start,
        'H_read_after_lock': True, 'model_selection': False, 'official_test_read': False,
        'warning': 'H is audit of frozen pilot only; viewing results then revising makes H development. Validation reused; sensitivity cohort is not guaranteed leakage-free.'})
    _commit(run, name, out)
    return pd.DataFrame(rows)


def show_audit_figures(run_id, cohort):
    """Display genuine cached results; never fit or create predictions here."""
    import matplotlib.pyplot as plt
    run, _, config = session(run_id)
    completed = _completed(run, '05_audit_' + cohort)
    if completed is None:
        raise ValueError('Audit trước khi vẽ.')
    existing = _completed(run, '06_figures_' + cohort)
    if existing is not None:
        from IPython.display import display, Image
        for path in sorted(existing.glob('*.png')):
            display(Image(filename=str(path)))
        return pd.read_csv(completed / 'metrics.csv')
    output = _attempt(run, '06_figures_' + cohort)
    table = pd.read_csv(completed / 'metrics.csv')
    bins = pd.read_csv(completed / 'calibration_bins.csv')
    for subset in table.cohort.unique():
        selected = table[(table.cohort == subset) & (table.model != 'additive_pooling_alias')]
        fig, axes = plt.subplots(4, 5, figsize=(16, 12), squeeze=False)
        for ax, row in zip(axes.flat, selected.itertuples()):
            cm = np.array([[row.cm_fake_fake, row.cm_fake_real], [row.cm_real_fake, row.cm_real_real]])
            ax.imshow(cm, cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center')
            ax.set(xticks=[0, 1], xticklabels=['fake', 'real'], yticks=[0, 1], yticklabels=['fake', 'real'],
                   xlabel='Dự đoán', ylabel='Nhãn thật', title=f'{row.model}\n{row.representation}')
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        fig.suptitle(f'{subset}: mọi baseline/pooling raw và calibrated; additive là alias')
        fig.tight_layout()
        fig.savefig(output / f'{subset}_confusion_all.png', dpi=140)
        plt.show()
    # One reliability panel per model, raw and calibrated on H only / original Val.
    subset = 'H_meta_audit' if cohort == 'H' else 'benchmark_original'
    reliability_names = config['base_models'] + ['fixed_pair_mean', 'normalized_pair_min', 'weighted_mean', 'learned_nonnegative_choquet']
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for ax, name in zip(axes.flat, reliability_names):
        for kind in ['raw', 'calibrated']:
            b = bins[(bins.cohort == subset) & (bins.model == name) & (bins.representation == kind) & (bins['mode'] == 'real_class') & (bins.n > 0)]
            if len(b):
                ax.plot(b.mean_probability, b.observed_frequency, 'o-', label=kind)
        ax.plot([0, 1], [0, 1], '--', color='grey')
        ax.set(title=name, xlabel='Trung bình p(real)', ylabel='Tỷ lệ real', xlim=(0, 1), ylim=(0, 1))
        ax.legend()
    fig.suptitle(f'{subset}: reliability bins cố định; ECE hữu hạn phụ thuộc bin và cỡ mẫu')
    fig.tight_layout()
    fig.savefig(output / f'{subset}_reliability.png', dpi=150)
    plt.show()
    print('Loss/F1 theo epoch: N/A cho pilot cổ điển; không dựng đường cong giả.')
    _commit(run, '06_figures_' + cohort, output)
    return table


def write_pilot_summary(run_id):
    run, _, _ = session(run_id)
    h, val = _completed(run, '05_audit_H'), _completed(run, '05_audit_Validation')
    if h is None or val is None:
        raise ValueError('Hoàn tất hai audit trước khi xuất báo cáo.')
    name = '07_summary'
    done = _completed(run, name)
    if done is not None:
        return done / 'pilot_report.md'
    out = _attempt(run, name)
    rows = pd.concat([pd.read_csv(folder / 'metrics.csv') for folder in [h, val]], ignore_index=True)
    write_csv(out / 'pilot_all_metrics.csv', rows)
    lines = [f'# Pilot H1/H2 — {run_id}', '',
        'Cấu hình pilot đã khóa trước H; chưa khóa mô hình nghiên cứu cuối. Test vẫn khóa.', '',
        'Base chỉ fit A; sigmoid fit C; combiner fit M. Không so ngang với Train-full lịch sử.', '',
        'H là audit của pipeline mới, không phải dữ liệu chưa từng dùng trong lịch sử. Nếu sửa theo H, H trở thành development.', '',
        'Validation dùng nhiều lần; cohort 2064 không được bảo đảm leakage-free. Không học tham số bằng Validation.', '',
        'Brier là binary real-class MSE; ECE top-label 10 bin cố định. SVM raw probability metrics N/A.', '',
        'Additive pooling là alias của weighted mean, không phải một estimator độc lập.', '',
        'Tất cả điểm là single-seed42; không 5-fold/multi-seed. Không tuyên bố tính mới hay chọn winner tự động.', '',
        'Không có epoch loss/F1 cho estimator cổ điển; A in-sample và M meta-fit không phải audit độc lập.', '']
    columns = ['model', 'representation', 'weighted_f1', 'macro_f1', 'accuracy', 'mcc', 'log_loss', 'brier_real_binary', 'ece_top_10', 'mce_top_10', 'ece_real_10']
    for cohort in rows.cohort.unique():
        lines += [f'## {cohort}', '', '| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
        for _, row in rows[rows.cohort == cohort].iterrows():
            values = [str(row[c]) if c in ['model', 'representation'] else 'N/A' if pd.isna(row[c]) else f'{row[c]:.6f}' for c in columns]
            lines.append('| ' + ' | '.join(values) + ' |')
        lines.append('')
    lines += ['## Artifact', '', 'Metrics toàn bộ, per-class và confusion counts: `pilot_all_metrics.csv`.', '',
        f'Assignment và nguồn: `{run.relative_to(ROOT)}/00_prepare_COMPLETE.json`.', '',
        f'Inference timing: `{h.relative_to(ROOT)}/timing.csv` và `{val.relative_to(ROOT)}/timing.csv`.', '',
        'Thời gian base-fit gồm TF-IDF và estimator; calibration-fit/combiner-fit ghi riêng ở marker tương ứng. Aggregation không phải end-to-end.', '',
        '**Dừng và gửi kết quả cho người nghiên cứu; chưa mở rộng kiến trúc hoặc phương pháp.**']
    with (out / 'pilot_report.md').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    _commit(run, name, out)
    return out / 'pilot_report.md'
