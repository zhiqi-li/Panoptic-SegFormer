import argparse
import os
from re import A
import warnings
import easymd
import mmcv
import torch
import os.path as osp
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from easymd.apis import multi_gpu_test_plus,single_gpu_test_plus,multi_gpu_test_plus2
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
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
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


Boot_up = False

def main(extra_args=None,checkpoint=None,path=None):
    args = parse_args()
    print(args.eval)
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if extra_args is not None:
        cfg.merge_from_dict(extra_args)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None
    #cfg.data.test.segmentations_folder= osp.join(args.config,'seg')
    # in case the test dataset is concatenated
    if path is not None:
        cfg.data.test.img_prefix = path
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        global Boot_up
        if not Boot_up:
            init_dist(args.launcher, **cfg.dist_params)
        Boot_up = True
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if checkpoint  == None:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        assert False,'TODO'
        outputs = single_gpu_test_plus(model, data_loader, args.show, args.show_dir,
                                  args.show_score_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        segmentations_folder = cfg.data.test.segmentations_folder
        datasets = cfg.model.bbox_head.get('datasets','coco')
        outputs = multi_gpu_test_plus2(model, data_loader,datasets,segmentations_folder, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            print(eval_kwargs)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(eval_kwargs)
            results = dataset.evaluate(outputs, **eval_kwargs)
            print(results)
            '''
            try:
                import wandb
                config = {
                    'learning_rate': cfg.optimizer.lr,
                    'epoch':cfg.runner.max_epochs,
                    'backbone':cfg.model.backbone.type,
                    'depth': cfg.model.backbone.get('depth','0'),
                    'bbox_head': cfg.model.bbox_head.type,
                    'num_query':cfg.model.bbox_head.num_query,
                    'transformer_head':cfg.model.bbox_head.transformer_head.type,
                    #'thre1': cfg.model.bbox_head.thre1,
                    #'thre2': cfg.model.bbox_head.thre2,
                }
                init_flag = False
                for kwarags in cfg.log_config.hooks:
                    if kwarags['type'] == 'WandbLoggerHook':
                        init_flag=True
                        wandb.init(config=config, **kwarags['init_kwargs'])
                        break
                if not init_flag:
                    wandb.init(config=config)
                tables = {}
                if 'All' in results.keys():
                    wandb_key_pano = ['type', 'pq','sq','rq']
                    wandb_data_pano = []
                    for key in {'All','Things','Stuff'}:
                        wandb_data_pano.append([key,results[key]['pq'],results[key]['sq'],results[key]['rq']])
                    table_panoptic = wandb.Table(data=wandb_data_pano,columns=wandb_key_pano)
                    tables['panptic'] = table_panoptic
                wandb_key = ['type','mAP','mAP_50','mAP_75','mAP_s','mAP_m','mAP_l']
                wandb_data = []
                for key in {'segm','bbox'}:
                    if  (key+'_mAP') in results.keys():
                        data = [key]
                        data.extend([results[key+'_'+each] for each in wandb_key[1:]])
                        wandb_data.append(data)
                if wandb_data !=[]:
                    det_table = wandb.Table(data=wandb_data,columns=wandb_key)
                    tables['det'] = det_table
                if tables !={}:
                    wandb.log(tables)
                wandb.join()
            except ImportError:
                print('Wandb not install')
            '''
import os
if __name__ == '__main__':
    
    files = sorted(os.listdir('/home/lzq/coco_c'))
    for file in files:
        
        if '_' in file:
            print('begin',file)
            path =os.path.join('/home/lzq/coco_c',file)
            main(path=path)
            print('end',file)
            #main(path)
    #for i in range(31,37):
    #    checkpoint = './work_dirs/pseg_r50_36e_detr/epoch_{i}.pth'.format(i=i)
    #    print(i)
    #    main(checkpoint=checkpoint)
    '''
    for thre1 in range(2,6,1):
        thre1/=10
        for thre2 in range(2,6,1):
            thre2/=10
            for use_argmax in [False]:
                for t1 in range(10,31,5):
                    t1/=100
                    for t2 in range(10,31,5):
                            t2 /=100
                            main(extra_args = 
                            {
                                'model.bbox_head.overlap_threshold1':thre1,
                                'model.bbox_head.thre1':t1,
                                'model.bbox_head.thre2':t2,
                                'model.bbox_head.overlap_threshold2':thre2,
                                'model.bbox_head.use_argmax':use_argmax})
                            args = {
                                'model.bbox_head.overlap_threshold1':thre1,
                                'model.bbox_head.thre1':t1,
                                'model.bbox_head.thre2':t2,
                                'model.bbox_head.overlap_threshold2':thre2,
                                'model.bbox_head.use_argmax':use_argmax}
                            print(args)
       '''                 
    