import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

from ml import FeatureConfig, build_features, build_labels, load_model
from train import (FEATURE_PARAMS, FOREST_PARAMS, HORIZON, PERIODS, TRAIN_END,
                   WORKERS, data_hash, read_prices)

config = FeatureConfig(**FEATURE_PARAMS)
cutoff = pd.Timestamp(TRAIN_END)
start, end = map(pd.Timestamp, PERIODS['validation'])
fingerprint = data_hash()
train_x, train_y, valid_x, valid_y = [], [], [], []
market = read_prices(Path('data/^N225_5m.csv'), end=end)
symbols = sorted(pd.read_csv('tickers.csv').symbol)
for i, symbol in enumerate(symbols):
    frame = read_prices(Path('data') / f'{symbol}_5m.csv', end=end)
    features = build_features(frame, market, config, as_of=end)
    labels = build_labels(frame, config, horizon=HORIZON, as_of=end)
    complete = features.notna().all(axis=1) & labels.target.notna()
    train = complete & (features.index <= cutoff) & labels.exit_time.le(cutoff)
    valid = complete & (features.index >= start) & labels.exit_time.le(end)
    train_x.append(features.loc[train])
    train_y.append(labels.loc[train, 'target'])
    valid_x.append(features.loc[valid])
    valid_y.append(labels.loc[valid, 'target'])
    if (i + 1) % 10 == 0 or i + 1 == len(symbols):
        print(f'prepared {i + 1}/{len(symbols)}', flush=True)
train_x, train_y = pd.concat(train_x), pd.concat(train_y).astype(int)
valid_x, valid_y = pd.concat(valid_x), pd.concat(valid_y).astype(int)
baseline = load_model('models/rf_current.pkl')
print(json.dumps({'train_rows': len(train_x), 'valid_rows': len(valid_x),
                  'train_up_share': float(train_y.mean()), 'workers': WORKERS}), flush=True)
for depth, leaf in [(5, 1), (10, 1), (10, 20), (20, 20), (20, 100), (None, 20)]:
    params = dict(FOREST_PARAMS, max_depth=depth, min_samples_leaf=leaf)
    classifier = RandomForestClassifier(**params, n_jobs=WORKERS)
    with threadpool_limits(limits=1):
        classifier.fit(train_x, train_y)
        p = classifier.predict_proba(valid_x)[:, 1]
        train_p = classifier.predict_proba(train_x)[:, 1]
    print(json.dumps({
        'depth': depth, 'leaf': leaf,
        'auc': float(roc_auc_score(valid_y, p)),
        'brier': float(brier_score_loss(valid_y, p)),
        'log_loss': float(log_loss(valid_y, p)),
        'train_auc': float(roc_auc_score(train_y, train_p)),
        'std': float(p.std()), 'min': float(p.min()), 'max': float(p.max()),
        'q05': float(np.quantile(p, .05)), 'q95': float(np.quantile(p, .95)),
        'constant_brier': float(brier_score_loss(valid_y, np.full(len(valid_y), train_y.mean()))),
    }), flush=True)
if data_hash() != fingerprint:
    raise ValueError('CSV changed during comparison')
