import os
import sys
import torch
import torch.nn as nn

script_dir = os.path.dirname(os.path.abspath(__file__))



from models import build_model
from models.csms6s_new import selective_scan_flop_jit
from config import get_config
import pdb

def main(config):    
    input_shape= [256, 256]
    if len(input_shape) == 1:
        input_shape = (3,input_shape[0], input_shape[0])
    elif len(input_shape) == 2:
        input_shape = (3, ) + tuple(input_shape)
    else:
        raise ValueError('invalid input shape')
    
    device = 'cuda'
    assert torch.cuda.is_available(
        ), 'No valid cuda device detected, please double check...'
    
    model = build_model(config)
    model.cuda()
    
    outputs=FLOPs.fvcore_flop_count(model,input_shape=input_shape)    
    print(outputs)
    split_line = '=' * 30
    batch_size=1
    input_shape = (batch_size, ) + input_shape
    print(f'{split_line}\nInput shape: {input_shape}\n'
          )    
    
def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def DCNv4Function_jit(inputs, outputs):
    if 1:
        print_jit_input_names(inputs)
    B, H, W, C = inputs[0].type().sizes()
    flops =36*H*W*C
    return flops

# used for print flops
class FLOPs:
    @staticmethod
    def register_supported_ops(model):
        if 0:
            supported_ops={
            "aten::gelu": None, # as relu is in _IGNORED_OPS
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit, # latter
            "prim::PythonOp.MambaInnerFnNoOutProj": partial(MambaInnerFnNoOutProj_flop_jit_vim, layer=model.backbone.layers[0]),
            # "aten::scaled_dot_product_attention": ...
        }
        else:
            supported_ops={
            "aten::gelu": None, # as relu is in _IGNORED_OPS
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit, # latter
            "prim::PythonOp.SelectiveScanCuda": selective_scan_flop_jit,
            "prim::PythonOp.DCNv4Function": DCNv4Function_jit,
        }
        return supported_ops

    @staticmethod
    def check_operations(model: nn.Module, inputs=None, input_shape=(3, 224, 224)):
        from fvcore.nn.jit_analysis import _get_scoped_trace_graph, _named_modules_with_dup, Counter, JitModelAnalysis
        
        if inputs is None:
            assert input_shape is not None
            if len(input_shape) == 1:
                input_shape = (1, 3, input_shape[0], input_shape[0])
            elif len(input_shape) == 2:
                input_shape = (1, 3, *input_shape)
            elif len(input_shape) == 3:
                input_shape = (1, *input_shape)
            else:
                assert len(input_shape) == 4
            inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)
        model.eval()
        flop_counter = JitModelAnalysis(model, inputs)
        flop_counter._ignored_ops = set()
        flop_counter._op_handles = dict()
        assert flop_counter.total() == 0 # make sure no operations supported
        print(flop_counter.unsupported_ops(), flush=True)
        print(f"supported ops {flop_counter._op_handles}; ignore ops {flop_counter._ignored_ops};", flush=True)

    @classmethod
    def fvcore_flop_count(cls, model: nn.Module, inputs=None, input_shape=(3, 224, 224), show_table=False, show_arch=False, verbose=True):
        supported_ops = cls.register_supported_ops(model)
        from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
        from fvcore.nn.flop_count import flop_count, FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS
        from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
        from fvcore.nn.jit_analysis import _IGNORED_OPS
        from fvcore.nn.jit_handles import get_shape, addmm_flop_jit
        
        if inputs is None:
            assert input_shape is not None
            if len(input_shape) == 1:
                input_shape = (1, 3, input_shape[0], input_shape[0])
            elif len(input_shape) == 2:
                input_shape = (1, 3, *input_shape)
            elif len(input_shape) == 3:
                input_shape = (1, *input_shape)
            else:
                assert len(input_shape) == 4

            inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)
        model.eval() 
        Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)
        params = fvcore_parameter_count(model)[""]
        flops = sum(Gflops.values())
        if verbose:
            print(Gflops.items())
            print("GFlops: ", flops, "Params: ", params, flush=True)        
        return params, flops

    # equals with fvcore_flop_count
    @classmethod
    def mmengine_flop_count(cls, model: nn.Module = None, input_shape = (3, 224, 224), show_table=False, show_arch=False, _get_model_complexity_info=False):
        supported_ops = cls.register_supported_ops()
        from mmengine.analysis.print_helper import is_tuple_of, FlopAnalyzer, ActivationAnalyzer, parameter_count, _format_size, complexity_stats_table, complexity_stats_str
        from mmengine.analysis.jit_analysis import _IGNORED_OPS
        from mmengine.analysis.complexity_analysis import _DEFAULT_SUPPORTED_FLOP_OPS, _DEFAULT_SUPPORTED_ACT_OPS
        from mmengine.analysis import get_model_complexity_info as mm_get_model_complexity_info
        def get_model_complexity_info(
            model: nn.Module,
            input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...],
                            None] = None,
            inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...],
                        None] = None,
            show_table: bool = True,
            show_arch: bool = True,
        ):
            if input_shape is None and inputs is None:
                raise ValueError('One of "input_shape" and "inputs" should be set.')
            elif input_shape is not None and inputs is not None:
                raise ValueError('"input_shape" and "inputs" cannot be both set.')

            if inputs is None:
                device = next(model.parameters()).device
                if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
                    inputs = (torch.randn(1, *input_shape).to(device), )
                elif is_tuple_of(input_shape, tuple) and all([
                        is_tuple_of(one_input_shape, int)
                        for one_input_shape in input_shape  # type: ignore
                ]):  # tuple of tuple of int, construct multiple tensors
                    inputs = tuple([
                        torch.randn(1, *one_input_shape).to(device)
                        for one_input_shape in input_shape  # type: ignore
                    ])
                else:
                    raise ValueError(
                        '"input_shape" should be either a `tuple of int` (to construct'
                        'one input tensor) or a `tuple of tuple of int` (to construct'
                        'multiple input tensors).')

            flop_handler = FlopAnalyzer(model, inputs).set_op_handle(**supported_ops)

            flops = flop_handler.total()
            params = parameter_count(model)['']
            flops_str = _format_size(flops)
            params_str = _format_size(params)

            if show_table:
                complexity_table = complexity_stats_table(
                    flops=flop_handler,
                    show_param_shapes=True,
                )
                complexity_table = '\n' + complexity_table
            else:
                complexity_table = ''

            if show_arch:
                complexity_arch = complexity_stats_str(
                    flops=flop_handler,
                )
                complexity_arch = '\n' + complexity_arch
            else:
                complexity_arch = ''

            return {
                'flops': flops,
                'flops_str': flops_str,
                'params': params,
                'params_str': params_str,
                'out_table': complexity_table,
                'out_arch': complexity_arch
            }
        
        if _get_model_complexity_info:
            return get_model_complexity_info

        model.eval()
        analysis_results = get_model_complexity_info(
            model,
            input_shape,
            show_table=show_table,
            show_arch=show_arch,
        )
        flops = analysis_results['flops_str']
        params = analysis_results['params_str']
        
        split_line = '=' * 30
        print(f'{split_line}\nInput shape: {input_shape}\t'
            f'Flops: {flops}\tParams: {params}\t'
        , flush=True)

    @classmethod
    def mmdet_flops(cls, config=None, extra_config=None):
        from mmengine.config import Config
        from mmengine.runner import Runner
        import numpy as np
        import os

        cfg = Config.fromfile(config)
        if "model" in cfg:
            if "pretrained" in cfg["model"]:
                cfg["model"].pop("pretrained")
        if extra_config is not None:
            new_cfg = Config.fromfile(extra_config)
            new_cfg["model"] = cfg["model"]
            cfg = new_cfg
        cfg["work_dir"] = "/tmp"
        cfg["default_scope"] = "mmdet"
        runner = Runner.from_cfg(cfg)
        model = runner.model.cuda()
        get_model_complexity_info = cls.mmengine_flop_count(_get_model_complexity_info=True)
        
        if True:
            oridir = os.getcwd()
            os.chdir(os.path.join(os.path.dirname(__file__), "../detection"))
            data_loader = runner.val_dataloader
            num_images = 100
            mean_flops = []
            for idx, data_batch in enumerate(data_loader):
                if idx == num_images:
                    break
                data = model.data_preprocessor(data_batch)
                model.forward = partial(model.forward, data_samples=data['data_samples'])
                out = get_model_complexity_info(model, input_shape=(3, 1280, 800))
                params = out['params_str']
                mean_flops.append(out['flops'])
            mean_flops = np.average(np.array(mean_flops))
            print(params, mean_flops)
            os.chdir(oridir)

    @classmethod
    def mmseg_flops(cls, config=None, input_shape=(3, 512, 2048)):
        from mmengine.config import Config
        from mmengine.runner import Runner

        cfg = Config.fromfile(config)
        cfg["work_dir"] = "/tmp"
        cfg["default_scope"] = "mmseg"
        runner = Runner.from_cfg(cfg)
        model = runner.model.cuda()
        
        cls.fvcore_flop_count(model, input_shape=input_shape)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, metavar="FILE", default="", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    main(config)



#python get_flops_mamba.py --cfg 'configs/hrvmamba/hrvmamba_nano_se_lpu.yaml'  