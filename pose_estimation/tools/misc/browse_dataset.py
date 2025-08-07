# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import mmcv
import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine import Config, DictAction
from mmengine.registry import build_from_cfg, init_default_scope
from mmengine.structures import InstanceData

from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples


from mmpose.registry import DATASETS, VISUALIZERS
from mmpose.structures import PoseDataSample
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument('--not-show', default=True, action='store_true')
    parser.add_argument(
        '--phase',
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=2,
        help='the interval of show (s)')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=3,
        help='Link thickness for visualization')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--mode',
        default='transformed',
        type=str,
        choices=['original', 'transformed'],
        help='display mode; display original pictures or transformed '
        'pictures. "original" means to show images load from disk'
        '; "transformed" means to show images after transformed;'
        'Defaults to "transformed".')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def generate_dup_file_name(out_file):
    """Automatically rename out_file when duplicated file exists.

    This case occurs when there is multiple instances on one image.
    """
    if out_file and osp.exists(out_file):
        img_name, postfix = osp.basename(out_file).rsplit('.', 1)
        exist_files = tuple(
            filter(lambda f: f.startswith(img_name),
                   os.listdir(osp.dirname(out_file))))
        if len(exist_files) > 0:
            img_path = f'{img_name}({len(exist_files)}).{postfix}'
            out_file = osp.join(osp.dirname(out_file), img_path)
    return out_file


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    backend_args = cfg.get('backend_args', dict(backend='local'))

    # register all modules in mmpose into the registries
    scope = cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    if args.mode == 'original':
        cfg[f'{args.phase}_dataloader'].dataset.pipeline = []
    else:
        # pack transformed keypoints for visualization
        cfg[f'{args.phase}_dataloader'].dataset.pipeline[
            -1].pack_transformed = True
    cfg[f'{args.phase}_dataloader'].dataset.bbox_file=None
    dataset = build_from_cfg(cfg[f'{args.phase}_dataloader'].dataset, DATASETS)

    visualizer = VISUALIZERS.build(cfg.visualizer)
    visualizer.set_dataset_meta(dataset.metainfo)

    #3###
    visualizer.line_width=3
    visualizer.radius=3
    ####


    progress_bar = mmengine.ProgressBar(len(dataset))

    ###
    cfg_options = None
    model = init_model(
        args.config,
        args.checkpoint,
        device=args.device,
        cfg_options=cfg_options)
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness
    ###

    idx = 0
    item = dataset[0]

    while idx < len(dataset):
        idx += 1
        next_item = None if idx >= len(dataset) else dataset[idx]

        if args.mode == 'original':
            if next_item is not None and item['img_path'] == next_item[
                    'img_path']:
                # merge annotations for one image
                item['keypoints'] = np.concatenate(
                    (item['keypoints'], next_item['keypoints']))
                item['keypoints_visible'] = np.concatenate(
                    (item['keypoints_visible'],
                     next_item['keypoints_visible']))
                item['bbox'] = np.concatenate(
                    (item['bbox'], next_item['bbox']))
                progress_bar.update()
                continue
            else:
                img_path = item['img_path']
                img_bytes = fileio.get(img_path, backend_args=backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='bgr')

                # forge pseudo data_sample
                gt_instances = InstanceData()
                gt_instances.keypoints = item['keypoints']
                gt_instances.keypoints_visible = item['keypoints_visible']
                gt_instances.bboxes = item['bbox']
                data_sample = PoseDataSample()
                data_sample.gt_instances = gt_instances

                item = next_item
        else:
            img = item['inputs'].permute(1, 2, 0).numpy()
            data_sample = item['data_samples']
            img_path = data_sample.img_path
            item = next_item

        out_file = osp.join(
            args.output_dir,
            osp.basename(img_path)) if args.output_dir is not None else None
        out_file = generate_dup_file_name(out_file)

        img = mmcv.bgr2rgb(img)

        # batch_results =print(data_sample,"aas",gt_instances.bboxes)
        batch_results = inference_topdown(model, img, bboxes=gt_instances.bboxes)
        results = merge_data_samples(batch_results)
        # print(results,"aa000")
        # skeleton_style ='openpose' skeleton_style = 'mmpose'

        visualizer.add_datasample(
            osp.basename(img_path),
            img,
            data_sample=results,
            draw_gt=False,
            draw_pred=True,
            draw_bbox=False,#(args.mode == 'original'),
            draw_heatmap=False,
            show=not args.not_show,
            wait_time=args.show_interval,
            skeleton_style ='mmpose',
            out_file=out_file)
        # pdb.set_trace()

        progress_bar.update()


if __name__ == '__main__':
    main()

# srun --partition=Gveval-S1 --quotatype=reserved --time=0-00:10:00 --mpi=pmi2 --gres=gpu:1 python tools/misc/browse_dataset.py configs/body_2d_keypoint/topdown_heatmap/cocofinal/td-hm_hrvmamba_base_8xb32-210e_coco-384x288_dcn_v17_a1.py work_dirs/td-hm_hrvmamba_base_8xb32-210e_coco-384x288_dcn_v17_a1_final_tip/best_coco_AP_epoch_204_final.pth --mode original --output-dir vis_results --phase val