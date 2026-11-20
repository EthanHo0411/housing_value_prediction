"""Task 1A result-format adapters over the shared training API.

Keep previously recorded JSON/checkpoints readable without duplicating training.
"""
from dataclasses import replace
import numpy as np
import torch
from housing.data import fold_split
from housing.mlp import MLP
from housing.scaling import Standardiser
from housing.train import Config, train_full_batch, cross_validate


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def config_from_saved(values):
    return Config(lr=values['learning_rate'], epochs=values['epochs'],
                  drop_rate=values['dropout'], loss=values['loss'], seed=values['seed'])


def saved_config(cfg):
    return dict(learning_rate=cfg.lr, epochs=cfg.epochs, dropout=cfg.drop_rate,
                loss=cfg.loss, seed=cfg.seed)


def saved_scaler(std):
    return dict(x_mean=std.mu_x.tolist(), x_std=std.sd_x.tolist(),
                y_mean=std.mu_y, y_std=std.sd_y, ddof=0)


def export_history(cfg, history, std, n_train, n_validation):
    return {'config': saved_config(cfg),
            'status': 'completed' if history.diverged_at is None else f'nonfinite_at_epoch_{history.diverged_at}',
            'curves': {'epoch': history.epoch, 'train_mae': history.train_mae,
                       'validation_mae': history.val_mae or [None] * len(history.epoch),
                       'standardised_loss': [None] + history.loss[:len(history.epoch)-1]},
            'scaler': saved_scaler(std), 'n_train': n_train, 'n_validation': n_validation}


def run_fold(cfg, fold=0):
    setup()
    train, val = fold_split(fold)
    fit = train_full_batch(cfg, train, val)
    return export_history(cfg, fit.history, fit.std, len(train), len(val))


def cross_validate_result(cfg):
    setup()
    # Preserve the recorded A3/A4 seed protocol without changing other tasks' defaults.
    cv = cross_validate(cfg, fold_seed_stride=1)
    folds = []
    for k, history in enumerate(cv.histories):
        if history.diverged_at is not None:
            raise RuntimeError(f'CV fold {k} diverged at epoch {history.diverged_at}')
        train, val = fold_split(k)
        result = export_history(replace(cfg, seed=cfg.seed+k), history,
                                Standardiser.fit(train), len(train), len(val))
        result['fold'] = k
        folds.append(result)
    mean, sd = cv.summary(ddof=1)
    return {'config': saved_config(cfg), 'folds': folds, 'fold_mae': cv.val_at().tolist(),
            'mean_mae': mean, 'std_mae': sd, 'std_ddof': 1, 'selection_epoch': cfg.epochs}


def load_checkpoint(path):
    """Read both the original Sequential checkpoint and the unified MLP format."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cfg = config_from_saved(checkpoint['config'])
    model = MLP(cfg.drop_rate)
    state = checkpoint['state_dict']
    if '0.weight' in state:
        mapping = {'0': 'fc1', '2': 'fc2', '4': 'out'}
        state = {mapping[key.split('.')[0]] + '.' + key.split('.')[1]: value
                 for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    s = checkpoint['scaler']
    std = Standardiser(np.array(s['x_mean']), np.array(s['x_std']), s['y_mean'], s['y_std'])
    return model, std, cfg
