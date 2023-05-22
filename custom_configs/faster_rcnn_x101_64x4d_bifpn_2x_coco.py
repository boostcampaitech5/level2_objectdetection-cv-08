_base_ = '../configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'
# custom_imports = dict(
#     imports=['mmdet.models.necks.bifpn.py'],
#     allow_failed_imports=False)
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
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    neck=dict(
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ))
