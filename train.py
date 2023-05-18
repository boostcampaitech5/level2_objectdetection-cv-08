from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import argparse
import wandb
from datetime import datetime



def main(args):
    set_random_seed(args.seed, deterministic=True)
    
    # classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    classes = ("Battery", "Clothing", "General trash", "Glass", "Metal", "Paper", "Paper pack", "Plastic", "Plastic bag", "Styrofoam")

    # config file 들고오기
    cfg = Config.fromfile(args.cfg)

    root='../../dataset/'

    # dataset config 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = "../../dataset/relabel/train"
    cfg.data.train.ann_file = "../../dataset/relabel/train/_annotations.coco.json"
    # cfg.data.train.pipeline[2]['img_scale'] = (512, 512) # Resize

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = "../../dataset/relabel/valid"
    cfg.data.val.ann_file = "../../dataset/relabel/valid/_annotations.coco.json"
    # cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    # cfg.data.test.pipeline[1]['img_scale'] = (512, 512) # Resize
    
    # batch size 수정
    cfg.data.samples_per_gpu = args.batch_size

    # seed
    cfg.seed = args.seed
    # gpu id
    cfg.gpu_ids = [0]
    # fp16
    cfg.fp16 = dict(loss_scale=512.)
    
    # 모델 weight, log 저장 위치
    today = datetime.now()
    today = today.strftime("%m-%d-%H:%M:%S")
    cfg_filename = args.cfg.split('/')[-1].split('.')[0]
    work_dir_name = cfg_filename + '_exp' + f'_{today}'
    cfg.work_dir = './work_dirs/' + work_dir_name

    # class 수 설정
    if cfg_filename == 'detectors_cascade_rcnn_r50_2x_coco' or cfg.model.backbone.type == 'mmcls.ConvNeXt' or cfg.model.backbone.type == 'SwinTransformer':
        cfg.model.roi_head.bbox_head[0].num_classes = 10 # cascade rcnn
        cfg.model.roi_head.bbox_head[1].num_classes = 10
        cfg.model.roi_head.bbox_head[2].num_classes = 10
    else:
        cfg.model.roi_head.bbox_head.num_classes = 10 # faster rcnn
    
    # cfg.model.bbox_head.num_classes = 10 

    cfg.model.test_cfg.rcnn.nms.iou_threshold = 0.6
    cfg.model.test_cfg.rcnn.max_per_image = 1000
    cfg.model.test_cfg.rcnn.score_thr = 0.0005

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=2, interval=1)
    cfg.device = get_device()
    print(cfg.data.train)
    
    # wandb logging
    if not args.no_wandb:
        cfg.log_config.hooks = [
            dict(type='TextLoggerHook'),
            dict(type='MMDetWandbHook',
                init_kwargs={'project': 'level2-hiboostcamp-2',
                            'entity': 'level2-hiboostcamp-2',
                            'name': f'jongmok_{cfg_filename}',
                            'config': {'optim': cfg.optimizer.type,
                                        'lr': cfg.optimizer.lr,
                                        'samples_per_gpu': cfg.data.samples_per_gpu,
                                        'cfg': args.cfg,
                                        'epoch': cfg.runner.max_epochs}
                            },
                interval=50,
                log_checkpoint=False,
                log_checkpoint_metadata=True,
                num_eval_images=0,
                bbox_score_thr=0.5)]
    
    datasets = [build_dataset(cfg.data.train)]
    print(datasets[0])
    
    model = build_detector(cfg.model)
    model.init_weights()
    
    meta = dict()
    train_detector(model, datasets[0], cfg, distributed=False, validate=True, meta=meta)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--cfg', type=str, default='./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help='path to load config file')
    parser.add_argument('--cfg', type=str, default='./custom_configs/baseline.py', help='path to load config file')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size(samples_per_gpu in MMDetection)')
    parser.add_argument('--no_wandb', action='store_true', default=False, help='disable wandb logging')
    
    args = parser.parse_args()
    
    main(args)