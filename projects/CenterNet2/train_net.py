import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.cpp_extension import CUDA_HOME
import time
import datetime
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tabulate import tabulate
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor

from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader

from centernet.config import add_centernet_config
from centernet.data.custom_build_augmentation import build_custom_augmentation

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' else \
            DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
            
        results[dataset_name]= inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results, evaluator

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume,
            ).get("iteration", -1) + 1
    )
    if cfg.SOLVER.RESET_ITER:
        logger.info('Reset loaded iteration. Start training from iteration 0.')
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )


    mapper = DatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
        DatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loader = build_detection_train_loader(cfg, mapper=mapper)
    else:
        from centernet.data.custom_dataset_dataloader import  build_custom_train_loader
        data_loader = build_custom_train_loader(cfg, mapper=mapper)


    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            loss_dict = model(data)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(args.config_file)[:-5]
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            logger.info("Running inference with test-time augmentation ...")
            model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)
        results, eval_instance = do_test(cfg, model)

        return results, eval_instance
        #return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )

    do_train(cfg, model, resume=args.resume)
    results, eval_instance = do_test(cfg, model)

    return results, eval_instance


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    #this has to be adjusted for each dataset parts:
    input_folder = "C:/Users/lillu/PycharmProjects/CenterNet2/WaymoCOCO/waymococo_rts/0004"
    #register coco instance -> adjust name according to the dataset part, and copy it to the configuration for DATASETS/TEST:(here,)
    register_coco_instances("waymo_rts_0004", {},
                            os.path.join(input_folder, 'annotations/image_info_test.json'),
                            os.path.join(input_folder, 'test'))
    args = default_argument_parser()
    args.add_argument('--manual_device', default='')
    args = args.parse_args()
    if args.manual_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.manual_device
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())
    print("Command Line Args:", args)
    result, eval_inst = launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    if eval_inst._coco_eval is not None:
        ious = eval_inst._coco_eval.ious
        gtruths = eval_inst._coco_eval._gts
        dets = eval_inst._coco_eval._dts
    else:
        ious, gtruths, dets = [0, 0, 0]
    cfg = setup(args)
    with open(os.path.join(cfg.OUTPUT_DIR, 'objs.pkl'), 'wb') as f:
        pickle.dump([ious, gtruths, dets, result],f)

    with open(os.path.join(cfg.OUTPUT_DIR, 'objs.pkl'), 'rb') as f:
        ious, gtruths, dets, result = pickle.load(f)
    print(result)
    path = os.path.join(input_folder, 'test/pred_boxes')
    os.makedirs(path)
    label_path = os.path.join(input_folder, 'annotations/image_info_test.json')
    g = open(label_path)
    out_dict = json.load(g)
    im_info = out_dict['images']
    for i in im_info:
        img_path1 = os.path.join(input_folder, 'test',i['file_name'])
        img = Image.open(img_path1)
        ax = plt.gca()
        ax.clear()
        plt.imshow(img)
        for j in dets:
            if j[0] == i['id']:
                for l in dets[j]:
                    x1 = l['bbox'][0]
                    y1 = l['bbox'][1]
                    w = l['bbox'][2]
                    h = l['bbox'][3]
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='tab:green', facecolor='none', linestyle='--', label=l['category_id'])
                    ax.add_patch(rect)
            else:
                continue
        for j in gtruths:
            if j[0] == i['id']:
                for l in gtruths[j]:
                    x1 = l['bbox'][0]
                    y1 = l['bbox'][1]
                    w = l['bbox'][2]
                    h = l['bbox'][3]
                    rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='tab:blue', facecolor='none', linestyle='-', label=l['category_id'])
                    ax.add_patch(rect)
            else:
                continue
        iou = []
        for j in ious:
            if j[0] == i['id']:
                for l in ious[j]:
                    iou.append(l)
        name = 'iou_table_%s' %i['id']
        with open(os.path.join(cfg.OUTPUT_DIR, 'iou_tables', '%s.txt' %name), 'w') as f:
            f.write(tabulate(iou))
        img_path2 = os.path.join(path, i['file_name'])
        plt.savefig(img_path2)
    g.close()


