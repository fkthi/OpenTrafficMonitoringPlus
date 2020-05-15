from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator
import os
import torch
import copy
import warnings
import argparse

warnings.simplefilter("ignore", FutureWarning)


def str_to_bool(v):
    if v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(
        "OpenTrafficMonitoring+ train Argument Parser")

    parser.add_argument('--weights',
                        default="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl",
                        type=str, help="Weights for training")
    parser.add_argument('--dataset_train', default="./custom_data/train/",
                        type=str,
                        help='Dataset folder used for training (should contain Images and .json annotations file)')
    parser.add_argument('--dataset_eval', default="./custom_data/eval/",
                        type=str,
                        help="Dataset folder used for evaluation every 'eval_interval' iterations (should also contain Images and .json annotations file)")
    parser.add_argument('--config',
                        default="./maskrcnn/mask_rcnn_R_50_FPN_1x.yaml",
                        type=str,
                        help='Detectron2 config file (not the OpenTrafficMonitoring+ config.py file !)')
    parser.add_argument('--workers', default=0, type=int,
                        help='Number of additional processed launched for the Pytorch Dataloader')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='Model batch size')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Model learning rate')
    parser.add_argument('--max_iter', default=30000, type=int,
                        help='Maximum Iterations')
    parser.add_argument('--eval_interval', default=10000, type=int,
                        help="Evaluation interval -> uses '--dataset_eval' for evaluation")
    parser.add_argument('--gamma', default=0.3, type=float,
                        help="Weight decay")
    parser.add_argument('--steps', default=(20000, 25000), type=int,
                        help="Learning rate steps. Pass argument as '--steps x y'  without '='",
                        nargs='+')
    parser.add_argument('--save_interval', default=5000, type=int,
                        help="Saves a model checkpoint every n steps")
    parser.add_argument('--warmup_iters', default=2000, type=int,
                        help="learning rate warmup steps")
    parser.add_argument('--freeze_at', default=2, type=int,
                        help="Freezes the Network at block x -> 0 to train all layers")
    parser.add_argument('--output_dir', default="./model_output/", type=str,
                        help="Output dir for the checkpoints and log files")
    parser.add_argument('--num_classes', default=1, type=int,
                        help="Number of Classes in the training set")
    parser.add_argument('--resume', default=False, type=str_to_bool)

    global args
    args = parser.parse_args()


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image, transforms = T.apply_transform_gens([
        T.Resize((1920, 1080)),
        T.RandomFlip(0.1),
        T.RandomSaturation(0.9, 1.1),
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1)
    ], image)

    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class CustomDefaultTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, True)


if __name__ == '__main__':
    parse_args()
    register_coco_instances(
        "dataset_train", {},
        args.dataset_train + "transformed_annotations.json",
        args.dataset_train)

    register_coco_instances(
        "dataset_test", {},
        args.dataset_eval + "transformed_annotations.json",
        args.dataset_eval)

    dataset_dicts_train = DatasetCatalog.get("dataset_train")
    meta_train = MetadataCatalog.get("dataset_train")

    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ("dataset_test",)
    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.MODEL.WEIGHTS = args.weights
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.OUTPUT_DIR = args.output_dir
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters
    cfg.TEST.EVAL_PERIOD = args.eval_interval
    cfg.SOLVER.GAMMA = args.gamma
    cfg.SOLVER.STEPS = tuple(args.steps)
    cfg.SOLVER.CHECKPOINT_PERIOD = args.save_interval
    cfg.MODEL.BACKBONE.FREEZE_AT = args.freeze_at
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (256)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = CustomDefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    trainer.train()
