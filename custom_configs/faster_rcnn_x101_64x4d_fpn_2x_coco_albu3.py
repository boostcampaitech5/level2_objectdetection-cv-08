_base_ = '../configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(
        type='HorizontalFlip',
        p=0.4),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0),
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0),
            dict(
                type="ToGray",
                p=1.0)],
        p=0.4),
    dict(
        type='IAAAdditiveGaussianNoise',
        p=0.3)
    ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0),
    dict(type='Pad', size_divisor=32),
    dict(type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg', 'pad_shape', 'scale_factor')),
    
]
data = dict(train=dict(pipeline=train_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=30)