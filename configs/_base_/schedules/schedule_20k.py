# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=20000)

eval_interval = 100
checkpoint_config = dict(by_epoch=False, interval=eval_interval)
evaluation = dict(
    interval=eval_interval,
    metric='mFscore',
    pre_eval=True
)
