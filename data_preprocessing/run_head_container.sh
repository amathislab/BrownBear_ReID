#!/bin/bash

mmdetection_root=/media/mu/bear/other_projects_mu/Object_detection_benchmarks/mmdetection
cd $mmdetection_root

export PYTHONPATH=$PYTHONPATH:$mmdetection_root

src_image_root=$BODY_CROPPED_IMAGES_DIR
dest_image_root=$HEAD_CROPPED_IMAGES_DIR

config=configs/faster_rcnn/faster_rcnn_r50_fpn_1x_bear_head.py
work_dir=work_dirs/animal_detection/faster_rcnn_r50_fpn_1x_bear_head
checkpoint=$work_dir/latest.pth

for bear_dir in $src_image_root/*  ## bear name
do
    bear_name=`basename $bear_dir`  ## BEAR NAME
    bear_output_dir=$dest_image_root/$bear_name

    for folder in $bear_dir/*  ### single or multiple
    do
        subfolder=`basename $folder`
        output_dir=$bear_output_dir/$subfolder
        src_dir=$folder
        python demo/bear_image_demo.py --image_folder $src_dir \
                                        --bear_name $bear_name \
                                        --config $config \
                                        --checkpoint $checkpoint \
                                        --out_folder $output_dir \
                                        --raw_csv $RAW_ANNOTATION_CSV \
                                        --body_csv $BODY_ANNOTATION_CSV
    done
done

echo '==============================================='