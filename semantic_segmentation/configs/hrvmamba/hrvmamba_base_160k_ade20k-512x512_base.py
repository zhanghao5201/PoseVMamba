_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='Backbone_HRVMamba',
        out_indices=(0, 1, 2, 3),
        pretrained = '/mnt/petrelfs/zhanghao.p/PoseVMamba/pretrain_model/hrvmamba_base_best.pth',
        extra=dict(
                    drop_path_rate=0.15,
                    stage1=dict(
                        num_modules=1,
                        num_branches=1,
                        block='BOTTLENECK',
                        num_blocks=(2, ),
                        num_channels=(64, ),
                        num_heads=[2],
                        mlp_ratios=[4]),
                    stage2=dict(
                        num_modules=1,
                        num_branches=2,
                        block='HRVmambaBlock',
                        num_blocks=(2, 2),
                        num_channels=(80, 160),
                        # =========================
                        ssm_d_state=1,
                        ssm_ratio=2.0,
                        ssm_dt_rank="auto",
                        ssm_act_layer="silu",        
                        ssm_conv=3,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0, 
                        ssm_init="v0",
                        forward_type="v05_noz",
                        # =========================
                        mlp_ratio=2.0,
                        mlp_act_layer="gelu",
                        mlp_drop_rate=0.0,
                        gmlp=False,
                        # =========================               
                        
                        ),
                    stage3=dict(
                        num_modules=4,
                        num_branches=3,
                        block='HRVmambaBlock',
                        num_blocks=(2, 2, 2),
                        num_channels=(80, 160, 320),
                        # =========================
                        ssm_d_state=1,
                        ssm_ratio=2.0,
                        ssm_dt_rank="auto",
                        ssm_act_layer="silu",        
                        ssm_conv=3,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0, 
                        ssm_init="v0",
                        forward_type="v05_noz",
                        # =========================
                        mlp_ratio=2.0,
                        mlp_act_layer="gelu",
                        mlp_drop_rate=0.0,
                        gmlp=False,
                        # =========================                 
                        ),
                    stage4=dict(
                        num_modules=2,
                        num_branches=4,
                        block='HRVmambaBlock',
                        num_blocks=(2, 2, 2, 2),
                        num_channels=(80, 160, 320, 640),
                        # =========================
                        ssm_d_state=1,
                        ssm_ratio=2.0,
                        ssm_dt_rank="auto",
                        ssm_act_layer="silu",        
                        ssm_conv=3,
                        ssm_conv_bias=False,
                        ssm_drop_rate=0.0, 
                        ssm_init="v0",
                        forward_type="v05_noz",
                        # =========================
                        mlp_ratio=2.0,
                        mlp_act_layer="gelu",
                        mlp_drop_rate=0.0,
                        gmlp=False,
                        # ========================= 
                        )),
        ),
    decode_head=dict(in_channels=[128,256,512,2048], num_classes=150),
    auxiliary_head=dict(in_channels=512, num_classes=150))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),##vmamba0.01 spatialmamba也是 0.00006
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# This model is trained on 2 nodes, 16 GPUs, 1 image per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
