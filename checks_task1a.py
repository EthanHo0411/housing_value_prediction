"""Integration checks for Task 1A; never read the test set or write results.

Use --reproduce for all recorded A1-A4 candidates (about a minute).
"""
import argparse
import json
import tempfile
import shutil
import io
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from housing.data import Rows, all_train, fold_split
from housing.train import Config, train_full_batch, predict_dollars, cross_validate, configure_runtime
from housing.mlp import load_checkpoint
from housing.report import fit_result, cv_result
from experiments import t1a5_final_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    configure_runtime()
    tr, va = fold_split(0)
    cfg = Config(lr=.001, epochs=4, drop_rate=.2, seed=42)
    a = train_full_batch(cfg, tr, va)
    b = train_full_batch(cfg, tr, Rows(va.x + 10000, va.y + 1e6))
    assert all(torch.equal(p, q) for p, q in zip(a.model.parameters(), b.model.parameters()))
    assert np.array_equal(a.std.mu_x, b.std.mu_x)
    c = train_full_batch(cfg, tr, va)
    assert a.history == c.history
    np.testing.assert_allclose(predict_dollars(a.model, a.std, va, batch=17),
                               predict_dollars(a.model, a.std, va), atol=.1, rtol=0)
    with patch.object(t1a5_final_test, 'load_test', side_effect=AssertionError('test access')), \
         patch.object(t1a5_final_test, 'train_full_batch', side_effect=AssertionError('retraining')):
        t1a5_final_test.main()
    model, std, cfg = load_checkpoint('results/t1a5_model.pt')
    rows = all_train()
    assert np.allclose(std.mu_x, rows.x.mean(0))
    pred = predict_dollars(model, std, rows)
    saved = json.loads(Path('results/t1a5_final_test.json').read_text())
    np.testing.assert_allclose(np.abs(pred-rows.y).mean(), saved['training']['curves']['train_mae'][-1], atol=.1, rtol=0)
    # Exercise a fresh A5 run with a training-fold stand-in, never the real test CSV.
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        for name in ['t1a3_cv_dropout.json', 't1a4_l1_loss.json']:
            shutil.copy2(Path('results') / name, output / name)
        captured = {}
        def capture(name, result):
            captured.update(result)
            return output / (name + '.json')
        with patch.object(t1a5_final_test, 'RESULTS', output), \
             patch.object(t1a5_final_test, 'load_test', return_value=va) as fake_test, \
             patch.object(t1a5_final_test, 'save_result', side_effect=capture), \
             redirect_stdout(io.StringIO()):
            t1a5_final_test.main()
        assert fake_test.call_count == 1 and captured['n_test'] == 100
        restored, restored_std, _ = load_checkpoint(output / 't1a5_model.pt')
        np.testing.assert_allclose(predict_dollars(restored, restored_std, rows), pred, atol=.1, rtol=0)
    print('PASS: validation isolation, deterministic training, batching, saved-test guard, both checkpoint formats, fresh A5 with stand-in data.')
    if args.reproduce:
        for name in ['t1a1_learning_rate','t1a2_dropout','t1a3_cv_dropout','t1a4_l1_loss']:
            recorded = json.loads(Path('results', name+'.json').read_text())
            gaps = []
            for old in recorded['runs']:
                cfg = Config.from_dict(old['config'])
                if 'folds' in old:
                    cv = cross_validate(cfg, fold_seed_stride=1)
                    new = cv_result(cv, ddof=1)
                    np.testing.assert_allclose(new['fold_mae'],old['fold_mae'],atol=1.,rtol=0)
                    assert new['std_ddof']==old['std_ddof']==1
                    gaps.append(max(abs(np.array(new['fold_mae'])-old['fold_mae'])))
                else:
                    fit = train_full_batch(cfg, tr, va)
                    new = fit_result(cfg, fit, len(tr), len(va))
                    assert new['status']==old['status']
                    for key in ['train_mae','validation_mae']:
                        np.testing.assert_allclose(new['curves'][key], old['curves'][key], atol=1.,rtol=1e-6)
                    gaps.append(max(abs(np.array(new['curves']['validation_mae'])-old['curves']['validation_mae'])))
            print(f'PASS {name}: maximum validation MAE difference ${max(gaps):.6f}',flush=True)


if __name__ == '__main__':
    main()
