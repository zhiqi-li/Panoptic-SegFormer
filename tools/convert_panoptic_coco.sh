#!/usr/bin/env bash

if [ $# == 0 ]
then
	echo "Need to at least one param to indicate which dataset to converte"
	return
fi

case $1 in

coco)
    python  $(dirname "$0")/converter.py  --input_json_file ./datasets/annotations/panoptic_val2017.json \
    --output_json_file ./datasets/annotations/panoptic_val2017_detection_format.json --categories_json_file \
    ./converter/panoptic_coco_categories.json

    python  $(dirname "$0")/converter.py  --input_json_file ./datasets/annotations/panoptic_train2017.json \
    --output_json_file ./datasets/annotations/panoptic_train2017_detection_format.json --categories_json_file \
    ./converter/panoptic_coco_categories.json
    ;;
cityscapes)
    python  $(dirname "$0")/converter.py  --input_json_file ./datasets/cityscapes/cityscapes_in_coco_format/cityscapes_panoptic_val.json \
    --output_json_file ./datasets/cityscapes/cityscapes_in_coco_format/cityscapes_panoptic_val_detection_format.json --categories_json_file \
    ./converter/panoptic_cityscapes_categories.json

    python  $(dirname "$0")/converter.py  --input_json_file ./datasets/cityscapes/cityscapes_in_coco_format/cityscapes_panoptic_train.json \
    --output_json_file ./datasets/cityscapes/cityscapes_in_coco_format/cityscapes_panoptic_train_detection_format.json --categories_json_file \
    ./converter/panoptic_cityscapes_categories.json
    ;;
ade20k)
    python  $(dirname "$0")/converter.py  --input_json_file ./datasets/ADEChallengeData2016/ade20k_panoptic_val.json \
    --output_json_file ./datasets/ADEChallengeData2016/ade20k_panoptic_val_detection_format.json --categories_json_file \
    ./converter/panoptic_ade20k_categories.json

    python  $(dirname "$0")/converter.py  --input_json_file  ./datasets/ADEChallengeData2016/ade20k_panoptic_train.json  \
    --output_json_file ./datasets/ADEChallengeData2016/ade20k_panoptic_train_detection_format.json  --categories_json_file \
    ./converter/panoptic_ade20k_categories.json
    ;;
mapillary)

    python  $(dirname "$0")/converter.py  --input_json_file ./datasets/mapillary-vistas/train/panoptic_2018.json \
    --segmentations_folder ./datasets/mapillary-vistas/train/panoptic  \
    --output_json_file ./datasets/mapillary-vistas/train/panoptic_train_coco_format.json --categories_json_file \
    ./converter/panoptic_mapillary.json

    python  $(dirname "$0")/converter.py  --input_json_file ./datasets/mapillary-vistas/val/panoptic_2018.json \
    --segmentations_folder ./datasets/mapillary-vistas/val/panoptic  \
    --output_json_file ./datasets/mapillary-vistas/val/panoptic_val_coco_format.json --categories_json_file \
    ./converter/panoptic_mapillary.json
esac
    
#gpu1 "python  ./tools/converter.py  --input_json_file ./datasets/cityscapes/cityscapes_in_coco_format/cityscapes_panoptic_train.json --output_json_file ./datasets/cityscapes/cityscapes_in_coco_format/cityscapes_panoptic_train_detection_format.json --categories_json_file ./converter/panoptic_cityscapes_categories.json"