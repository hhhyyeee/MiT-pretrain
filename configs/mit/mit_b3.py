_base_ = [
    '../_base_/models/mit.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256_adamw.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    pretrained='pretrained/mit_b3.pth',
    backbone=dict(
        type='mit_b3',
        style='pytorch',
        pet_cls='Adapter',
        adapt_blocks=[0, 1, 2, 3],
        aux_classifier=True),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        # in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False),
    train_cfg=dict(augments=dict(type='Mixup', alpha=0.2)),
)

param_scheduler = dict(
    type='CosineAnnealingLR', by_epoch=True, T_max=295, bigin=5, end=300)

backbone_freeze = True
