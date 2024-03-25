_base_ = [
    '../_base_/models/mit.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    pretrained='pretrained/mit_b5.pth',
    backbone=dict(
        type='mit_b5',
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

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    # optimizer=dict(type='Adam', lr=0.001, momentum=0.9, weight_decay=0.0001)
)

backbone_freeze = True
