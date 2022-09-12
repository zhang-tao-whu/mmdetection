custom_imports = dict(
    imports=['mmdet.models.plugins.custom_transformer'],
    allow_failed_imports=False)

_base_ = 'e2ec_fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_feature64_p4init_p4evolve_dml_1x_coco.py'

model = dict(
    type='LineFormer',
    line_pred_head=dict(
        type='LineFormerHead',
        in_channel=64,
        out_channel=4,
        num_query=30,
        roi_wh=(40, 40),
        expand_scale=1.1,
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=(40, 40), sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4]),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        line_predictor=dict(
            type='LinePredictor',
            line_decoder=dict(
                type='LineDecoder',
                return_intermediate=True,
                num_layers=3,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=256,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        line_loss=dict(
            type='LineCriter',
            score_weight=1.,
            line_weight=5.,
            point_weight=1.
        )),
    line_pred_start_level=0, # start from P2
)

# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(736, 256), (736, 608)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='AlignSampleBoundaryLine', point_nums=128, reset_bbox=True),
    dict(type='ContourLineDefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels',
         'gt_masks', 'gt_polys', 'key_points_masks', 'key_points',
         'gt_lines', 'num_lines']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(736, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
optimizer_config = dict(_delete_=True, grad_clip=None)

lr_config = dict(warmup='linear')
