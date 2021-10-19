# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/data/cropped'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (512, 512)

classes = ['background', 'road']
img_suffix = '.png'
seg_map_suffix = '.png'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        classes=classes,
        img_dir='images',
        img_suffix=img_suffix,
        ann_dir='masks/road',
        seg_map_suffix=seg_map_suffix,
        split='train_with_positive_pixels.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        classes=classes,
        img_dir='images',
        img_suffix=img_suffix,
        ann_dir='masks/road',
        seg_map_suffix=seg_map_suffix,
        split='test_with_positive_pixels.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        classes=classes,
        img_dir='images',
        img_suffix=img_suffix,
        ann_dir='masks/road',
        seg_map_suffix=seg_map_suffix,
        split='data/data/cropped/test_with_positive_pixels.txt',
        pipeline=test_pipeline))
