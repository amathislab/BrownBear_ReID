#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

DATAROOT= #Your data root path

RAW_IMAGE_DIR=$DATAROOT/raw  ### raw images

BODY_IMAGES_DIR=$DATAROOT/body  ### cropped body images
### if not exists, create it
if [ ! -d $BODY_IMAGES_DIR ]; then
    mkdir -p $BODY_IMAGES_DIR
fi

HEAD_IMAGES_DIR=$DATAROOT/head   ### cropped head images
### if not exists, create it
if [ ! -d $HEAD_IMAGES_DIR ]; then
    mkdir -p $HEAD_IMAGES_DIR
fi

ANNOTATION_DIR=$DATAROOT/annotation
### if not exists, create it
if [ ! -d $ANNOTATION_DIR ]; then
    mkdir -p $ANNOTATION_DIR
fi

RAW_ANNOTATION_CSV=$ANNOTATION_DIR/raw.csv
BODY_ANNOTATION_CSV=$ANNOTATION_DIR/body.csv
HEAD_ANNOTATION_CSV=$ANNOTATION_DIR/head.csv
BODY_DETECTION_OUTPUT_DIR=$BODY_IMAGES_DIR/detection_output   ### cameratrap outputs
BODY_CROPPED_IMAGES_DIR=$BODY_IMAGES_DIR/images_uncurated
HEAD_CROPPED_IMAGES_DIR=$HEAD_IMAGES_DIR/images_uncurated


##################################### GET RAW CSV #####################################
echo 'Start getting raw csv'

# check if raw csv exists
if [ -f "$RAW_ANNOTATION_CSV" ]
then
    echo "Raw CSV already exists"
else
    echo "Raw CSV does not exist, running raw csv"
    ### only run when RAW_IMAGE_DIR is not empty
    if [ -n "$(ls -A $RAW_IMAGE_DIR)" ]; then
        python data_analysis/preprocessing/get_raw_csv.py --image_dir ${RAW_IMAGE_DIR} \
                                                          --save_path ${RAW_ANNOTATION_CSV}
    else
        echo "RAW_IMAGE_DIR is empty, skipping raw csv"
        exit 1
    fi
fi

echo '==============================================='

# ##################################### RUN BODY DETECTOR #####################################
cd CameraTraps
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cameratraps-detector


for dir in $RAW_IMAGE_DIR/*
do
    filename=`basename $dir`  ## BEAR NAME
    output_dir=$BODY_DETECTION_OUTPUT_DIR/output_$filename.json
    ### check if output_file exists
    if [ -f $output_dir ]; then
        echo "Output file '$filename' already exists, skipping"
    else
        echo "Output file '$filename' does not exist, running body detector"
        python detection/run_tf_detector_batch.py ./md_v4.1.0.pb ${dir} --output_file $output_dir --recursive --threshold 0.90 --checkpoint_frequency -1
    fi

done

echo '==============================================='

cd ../
### only run when all body detection is done
### if ls $BODY_DETECTION_OUTPUT_DIR/*.json is not empty, run body csv
if [ -n "$(ls -A $BODY_DETECTION_OUTPUT_DIR/*.json)" ]; then
    # check if raw csv exists
    if [ -f $BODY_ANNOTATION_CSV ]; then
        echo "Body CSV already exists"
    else
        echo "All body detection is done, running body csv"
        
        python data_analysis/preprocessing/get_body_csv.py --body_detection_path ${BODY_DETECTION_OUTPUT_DIR} \
                                                            --output_dir ${BODY_CROPPED_IMAGES_DIR} \
                                                            --raw_csv_path ${RAW_ANNOTATION_CSV}
        echo "Body csv is done"
    fi
fi

echo '==============================================='

##################################### RUN HEAD DETECTOR #####################################

echo "Running head detector"

### check if head cropped images dir exists
if [ -d $HEAD_CROPPED_IMAGES_DIR ]; then
    echo "$HEAD_CROPPED_IMAGES_DIR exists"
else
    echo "$HEAD_CROPPED_IMAGES_DIR does not exist, running head detector"

    docker run -it \
        --gpus all \
        -v /media:/media \
        -v $PWD:/workspace \
        -e BODY_CROPPED_IMAGES_DIR=$BODY_CROPPED_IMAGES_DIR \
        -e HEAD_CROPPED_IMAGES_DIR=$HEAD_CROPPED_IMAGES_DIR \
        -e RAW_ANNOTATION_CSV=$RAW_ANNOTATION_CSV \
        -e BODY_ANNOTATION_CSV=$BODY_ANNOTATION_CSV \
        mmpose \
        bash run_head_container.sh
fi

echo '==============================================='

# ##################################### GET HEAD CSV (uncurated or curated) #####################################
echo 'Start getting head csv'

### replace 'uncurated' with 'curated' in $HEAD_CROPPED_IMAGES_DIR
CURATED_HEAD_CROPPED_IMAGES_DIR=${HEAD_CROPPED_IMAGES_DIR/uncurated/curated}

### check if HEAD_CROPPED_IMAGES_DIR exists, but not CURATED_HEAD_CROPPED_IMAGES_DIR
if [ -d $HEAD_CROPPED_IMAGES_DIR ] && [ ! -d $HEAD_ANNOTATION_CSV ]; then
    echo "$HEAD_CROPPED_IMAGES_DIR exists, but $HEAD_ANNOTATION_CSV does not exist, get uncurated csv file"

    python data_analysis/preprocessing/get_head_csv.py --head_cropped_images_dir $HEAD_CROPPED_IMAGES_DIR \
                                                       --head_csv_path $HEAD_ANNOTATION_CSV \
                                                       --merge_meta_info

    echo '==============================================='
fi

### if both $HEAD_ANNOTATION_CSV and CURATED_HEAD_CROPPED_IMAGES_DIR exists, run get_head_csv.py
if [ -f $HEAD_ANNOTATION_CSV ] && [ -d $CURATED_HEAD_CROPPED_IMAGES_DIR ]; then
    echo "$HEAD_ANNOTATION_CSV and $CURATED_HEAD_CROPPED_IMAGES_DIR exist, get curated csv file"

    python data_analysis/preprocessing/get_head_csv.py --head_cropped_images_dir $CURATED_HEAD_CROPPED_IMAGES_DIR \
                                                       --head_csv_path $HEAD_ANNOTATION_CSV \
                                                       --is_curated \
                                                       --merge_meta_info

    echo '==============================================='
fi

echo '=======================END========================'

