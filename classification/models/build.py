from .hrvmamba import HRVmamba
from .hrvmamba_se_lpu import HRVmamba_se_lpu
from .hrformer import HRFormer

def build_model(config):
    model_type = config.MODEL.TYPE    
    if model_type in ["hrvmamba_base"]:
        model = HRVmamba(in_channels=3,
                    norm_cfg=dict(type='BN'),
                    extra=dict(
                    drop_path_rate=0.2,
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
                        )))
        return model    
    if model_type in ["hrvmamba_small"]:
        model = HRVmamba(in_channels=3,
                    norm_cfg=dict(type='BN'),
                    extra=dict(
                    drop_path_rate=0.2,
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
                        num_channels=(32, 64),
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
                        gmlp=False   
                        ),
                    stage3=dict(
                        num_modules=4,
                        num_branches=3,
                        block='HRVmambaBlock',
                        num_blocks=(2, 2, 2),
                        num_channels=(32, 64, 128),
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
                        num_channels=(32, 64, 128, 256),
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
                        )))
        return model    
    if model_type in ["hrvmamba_tiny"]:
        model = HRVmamba_se_lpu(in_channels=3,
                    norm_cfg=dict(type='BN'),
                    extra=dict(
                    drop_path_rate=0.1,
                    stage1=dict(
                        num_modules=1,
                        num_branches=1,
                        block='BOTTLENECK',
                        num_blocks=(2, ),
                        num_channels=(32, ),
                        num_heads=[2],
                        mlp_ratios=[4]),
                    stage2=dict(
                        num_modules=1,
                        num_branches=2,
                        block='HRVmambaBlock',
                        num_blocks=(2, 2),
                        num_channels=(16, 32),
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
                        num_channels=(16, 32, 64),
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
                        num_channels=(16,32, 64, 128),
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
                        )))
        return model
    if model_type in ["hrvmamba_nano"]:
        model = HRVmamba_se_lpu(in_channels=3,
                    norm_cfg=dict(type='BN'),
                    extra=dict(
                    drop_path_rate=0.1,
                    stage1=dict(
                        num_modules=1,
                        num_branches=1,
                        block='BOTTLENECK',
                        num_blocks=(2, ),
                        num_channels=(16, ),
                        num_heads=[2],
                        mlp_ratios=[4]),
                    stage2=dict(
                        num_modules=1,
                        num_branches=2,
                        block='HRVmambaBlock',
                        num_blocks=(2, 2),
                        num_channels=(8, 16),
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
                        ),
                    stage3=dict(
                        num_modules=4,
                        num_branches=3,
                        block='HRVmambaBlock',
                        num_blocks=(2, 2, 2),
                        num_channels=(8, 16,32),
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
                        num_channels=(8,16,32, 64),
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
                        )))
        return model
    if model_type in ["hrformer_base"]:
        model = HRFormer(in_channels=3,
                    # norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_cfg=dict(type='BN'),
                extra=dict(
                drop_path_rate=0.2,
                with_rpe=True,
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(2, ),
                    num_channels=(64, ),
                    num_heads=[2],
                    num_mlp_ratios=[4]),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2),
                    num_channels=(78, 156),
                    num_heads=[2, 4],
                    mlp_ratios=[4, 4],
                    window_sizes=[7, 7]),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2, 2),
                    num_channels=(78, 156, 312),
                    num_heads=[2, 4, 8],
                    mlp_ratios=[4, 4, 4],
                    window_sizes=[7, 7, 7]),
                stage4=dict(
                    num_modules=2,
                    num_branches=4,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2, 2, 2),
                    num_channels=(78, 156, 312, 624),
                    num_heads=[2, 4, 8, 16],
                    mlp_ratios=[4, 4, 4, 4],
                    window_sizes=[7, 7, 7, 7])),
                init_cfg=None,)

        return model
    if model_type in ["hrformer_small"]:
        model = HRFormer(in_channels=3,
                    # norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_cfg=dict(type='BN'),
                extra=dict(
                drop_path_rate=0.2,
                with_rpe=True,
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(2, ),
                    num_channels=(64, ),
                    num_heads=[2],
                    num_mlp_ratios=[4]),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2),
                    num_channels=(32, 64),
                    num_heads=[1, 2],
                    mlp_ratios=[4, 4],
                    window_sizes=[7, 7]),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2, 2),
                    num_channels=(32, 64, 128),
                    num_heads=[1, 2, 4],
                    mlp_ratios=[4, 4, 4],
                    window_sizes=[7, 7, 7]),
                stage4=dict(
                    num_modules=2,
                    num_branches=4,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2, 2, 2),
                    num_channels=(32, 64, 128, 256),
                    num_heads=[1, 2, 4, 8],
                    mlp_ratios=[4, 4, 4, 4],
                    window_sizes=[7, 7, 7, 7])),
                init_cfg=None,)
        return model
    if model_type in ["hrformer_tiny"]:
        model = HRFormer(in_channels=3,
                    # norm_cfg=dict(type='SyncBN', requires_grad=True),
                norm_cfg=dict(type='BN'),
                extra=dict(
                drop_path_rate=0.1,
                with_rpe=True,
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(2, ),
                    num_channels=(32, ),
                    num_heads=[2],
                    num_mlp_ratios=[4]),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2),
                    num_channels=(16, 32),
                    num_heads=[1, 2],
                    mlp_ratios=[4, 4],
                    window_sizes=[7, 7]),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2, 2),
                    num_channels=(16, 32, 64),
                    num_heads=[1, 2, 4],
                    mlp_ratios=[4, 4, 4],
                    window_sizes=[7, 7, 7]),
                stage4=dict(
                    num_modules=2,
                    num_branches=4,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2, 2, 2),
                    num_channels=(16,32, 64, 128),
                    num_heads=[1, 2, 4, 8],
                    mlp_ratios=[4, 4, 4, 4],
                    window_sizes=[7, 7, 7, 7])),
                init_cfg=None,)
        return model
    if model_type in ["hrformer_nano"]:
        model = HRFormer(in_channels=3,
                norm_cfg=dict(type='BN'),
                extra=dict(
                drop_path_rate=0.1,
                with_rpe=True,
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=(2, ),
                    num_channels=(16, ),
                    num_heads=[2],
                    num_mlp_ratios=[4]),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2),
                    num_channels=(8, 16),
                    num_heads=[1, 2],
                    mlp_ratios=[4, 4],
                    window_sizes=[7, 7]),
                stage3=dict(
                    num_modules=4,
                    num_branches=3,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2, 2),
                    num_channels=(8, 16,32),
                    num_heads=[1, 2, 4],
                    mlp_ratios=[4, 4, 4],
                    window_sizes=[7, 7, 7]),
                stage4=dict(
                    num_modules=2,
                    num_branches=4,
                    block='HRFORMERBLOCK',
                    num_blocks=(2, 2, 2, 2),
                    num_channels=(8, 16,32,64),
                    num_heads=[1, 2, 4, 8],
                    mlp_ratios=[4, 4, 4, 4],
                    window_sizes=[7, 7, 7, 7])),
                init_cfg=None,)
        return model
    return None