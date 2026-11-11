"""Optimizer-update warmup and validation-only selection, without GPU imports."""
import math


def schedule(train_rows, batch_size, accumulation, epochs, warmup_epochs, peak_lr):
    if (train_rows < 2 or batch_size < 2 or batch_size % 2 or accumulation < 1
            or epochs < 1 or not math.isfinite(warmup_epochs)
            or not 0 < warmup_epochs <= epochs or not math.isfinite(peak_lr) or peak_lr <= 0):
        raise ValueError('Invalid training schedule')
    batches = (train_rows + batch_size - 1) // batch_size
    updates = (batches + accumulation - 1) // accumulation
    warmup = max(1, math.ceil(warmup_epochs * updates))
    return dict(name='linear_warmup_then_constant', train_rows=train_rows,
                microbatches_per_epoch=batches, updates_per_epoch=updates,
                total_updates=epochs*updates, warmup_epochs=warmup_epochs,
                warmup_updates=warmup, start_factor=.05, peak_lr=peak_lr)


def learning_rate(config, update):
    if not 1 <= update <= config['total_updates']:
        raise ValueError('Optimizer update outside frozen schedule')
    length = config['warmup_updates']
    fraction = min(1., (update - 1)/(length - 1)) if length > 1 else 1.
    return config['peak_lr'] * (config['start_factor'] + (1-config['start_factor'])*fraction)


def choose_validation(history, epochs):
    if len(history) != epochs or [r['epoch'] for r in history] != list(range(1, epochs+1)):
        raise ValueError('Incomplete training history')
    for record in history:
        if (not all(math.isfinite(record[k]) for k in ('train_loss', 'validation_loss',
                                                      'validation_auc', 'validation_score_std'))
                or record['train_loss'] < 0 or record['validation_loss'] < 0
                or not .55 < record['validation_auc'] <= 1
                or record['validation_score_std'] < 1e-4):
            raise ValueError('Validation learning gate failed; no test evaluation')
    selected = min(history, key=lambda r: r['validation_loss'])
    if selected['validation_loss'] >= math.log(2):
        raise ValueError('Selected validation CE does not beat constant 0.5; no test evaluation')
    return selected
