_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_2x.py', '../configs/_base_/default_runtime.py'
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='Albu',
         transforms=[
             dict(
                 type='HorizontalFlip',
                 p=0.2
             ),
             dict(
                 type='VerticalFlip',
                 p=0.2
             ),
            # dict(
            #     type='HueSaturationValue',
            #     hue_shift_limit=20,
            #     sat_shift_limit=30,
            #     val_shift_limit=20,
            #     p=0.2
            # ),
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.2
            ),
            dict(
                type='IAAAdditiveGaussianNoise',
                p=0.2
            )
         ],
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
        },),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
