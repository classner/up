#!/bin/bash

trap "exit" INT

CAFFE_BUILD_FP=$(../config.py DEEPLAB_BUILD_FP)
CAFFE_BIN=${CAFFE_BUILD_FP}/install/bin/caffe
CAFFE_PYTHON=${CAFFE_BUILD_FP}/install/python
EXP=training
if [ "${EXP}" = "training" ]; then
    DATA_ROOT=$(../config.py SEG_DATA_FP)
else
    echo "Wrong exp name"
    exit 1
fi

## Specify which model to train
########### training (standard training) ################
NET_ID=$2
TRAIN_SET_SUFFIX=$3
RESUME=$(grep -q resume <<<$4 && echo 1 || echo 0)
DEV_ID=0
RUN_TRAIN=`grep -q train <<<$1 && echo 1 || echo 0`
RUN_TEST=`grep -q test <<<$1 && echo 1 || echo 0`
RUN_EVALUATION=`grep -q evaluate <<<$1 && echo 1 || echo 0`
RUN_TRAIN2=`grep -q trfull <<<$1 && echo 1 || echo 0`
RUN_TEST2=`grep -q tefull <<<$1 && echo 1 || echo 0`
RUN_EVALUATION2=`grep -q evfull <<<$1 && echo 1 || echo 0`
#####
## Create dirs
CONFIG_DIR=${EXP}/config/${NET_ID}
MODEL_DIR=${EXP}/model/${NET_ID}
mkdir -p ${MODEL_DIR}
LOG_DIR=${EXP}/log/${NET_ID}
mkdir -p ${LOG_DIR}
export GLOG_log_dir=${LOG_DIR}

## Training #1 (on train_aug)
if [ ${RUN_TRAIN} -eq 1 ]; then
    export GLOG_minloglevel=0
    LIST_DIR=${EXP}/list
    NUM_CLASSES=`cat ${EXP}/config/${NET_ID}/n_classes.txt`
    ((NUM_LABELS=NUM_CLASSES+1))  # Add background.
    TARGET_PERSON_SIZE=`cat ${EXP}/config/${NET_ID}/target_person_size.txt`
    TRAIN_SET=train_${NUM_CLASSES}_${TARGET_PERSON_SIZE}_${TRAIN_SET_SUFFIX}
    if [ $RESUME -gt 0 ]; then
        MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
        SSTATE=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.solverstate | head -n 1`
        if [ ! -f ${SSTATE} ]; then
            echo "Solverstate not found: ${SSTATE}!"
            exit 1
        fi
    else
        MODEL=${EXP}/config/${NET_ID}/init.caffemodel
    fi
    if [ ! -f ${MODEL} ]; then
        echo "Initialization model file not found: ${MODEL}!"
        exit 1
    fi
    echo Training net ${EXP}/${NET_ID}
    for pname in train solver; do
        sed "$(eval echo $(cat sub.sed))" \
            ${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
    CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
    if [ -f ${MODEL} -a ${RESUME} -le 0 ]; then
        CMD="${CMD} --weights=${MODEL}"
    fi
    if [ ${RESUME} -gt 0 ]; then
        CMD="${CMD} --snapshot=${SSTATE}"
    fi
    echo Running ${CMD} && ${CMD}
fi

## Test #1 specification (on val or test)
if [ ${RUN_TEST} -eq 1 ]; then
    export GLOG_minloglevel=1
    for TEST_SET in val; do
        NUM_CLASSES=`cat ${EXP}/config/${NET_ID}/n_classes.txt`
        ((NUM_LABELS=NUM_CLASSES+1))  # Add background.
        TARGET_PERSON_SIZE=`cat ${EXP}/config/${NET_ID}/target_person_size.txt`
        TEST_SET=${TEST_SET}_${NUM_CLASSES}_${TARGET_PERSON_SIZE}_${TRAIN_SET_SUFFIX}
        TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
        MODEL=${EXP}/model/${NET_ID}/test.caffemodel
        if [ ! -f ${MODEL} ]; then
            MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
        fi
        if [ ! -f ${MODEL} ]; then
            echo "Test model file not found: ${MODEL}!"
            exit 1
        fi
        echo Testing net ${EXP}/${NET_ID}
        FEATURE_DIR=${EXP}/features/${NET_ID}
				mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
        sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/testpy.prototxt > ${CONFIG_DIR}/testpy_${TEST_SET}.prototxt
        CMD="./store_segmentation_results.py \
             ${CONFIG_DIR}/testpy_${TEST_SET}.prototxt \
             ${MODEL} \
             ${DATA_ROOT} \
             ${EXP}/list/${TEST_SET}.txt \
             ${FEATURE_DIR}/${TEST_SET}/seg_score \
             --caffe_install_path ${CAFFE_PYTHON} \
             --n_labels ${NUM_LABELS}"
        echo Running ${CMD} && ${CMD}
    done
fi

## Evaluation #1 specification (on val or test)
if [ ${RUN_EVALUATION} -eq 1 ]; then
    export GLOG_minloglevel=1
    for TEST_SET in val; do
        echo Evaluating net ${EXP}/${NET_ID}
        FEATURE_DIR=${EXP}/features/${NET_ID}
        NUM_CLASSES=`cat ${EXP}/config/${NET_ID}/n_classes.txt`
        ((NUM_LABELS=NUM_CLASSES+1))  # Add background.
        TARGET_PERSON_SIZE=`cat ${EXP}/config/${NET_ID}/target_person_size.txt`
        TEST_SET=${TEST_SET}_${NUM_CLASSES}_${TARGET_PERSON_SIZE}_${TRAIN_SET_SUFFIX}
        CMD="./evaluate_segmentation.py \
             ${EXP}/list/${TEST_SET}.txt \
             ${DATA_ROOT} \
             ${FEATURE_DIR}/${TEST_SET}/seg_score \
             ${NUM_LABELS}"
        echo Running ${CMD} && ${CMD}
    done
fi

################################################################################
## Training #2 (finetune on trainval_aug)
if [ ${RUN_TRAIN2} -eq 1 ]; then
    export GLOG_minloglevel=0
    LIST_DIR=${EXP}/list
    NUM_CLASSES=`cat ${EXP}/config/${NET_ID}/n_classes.txt`
    ((NUM_LABELS=NUM_CLASSES+1))
    TARGET_PERSON_SIZE=`cat ${EXP}/config/${NET_ID}/target_person_size.txt`
    TRAIN_SET=trainval_${NUM_CLASSES}_${TARGET_PERSON_SIZE}_${TRAIN_SET_SUFFIX}
    if [ $RESUME -gt 0 ]; then
        MODEL=`ls -t ${EXP}/model/${NET_ID}/train2_iter_*.caffemodel | head -n 1`
        SSTATE=`ls -t ${EXP}/model/${NET_ID}/train2_iter_*.solverstate | head -n 1`
        if [ ! -f ${SSTATE} ]; then
            echo "Solverstate not found: ${SSTATE}!"
            exit 1
        fi
    else
        MODEL=${EXP}/config/${NET_ID}/init2.caffemodel
    fi
    if [ ! -f ${MODEL} ]; then
        echo "WARNING! No init2 model file found at: ${MODEL}. Using latest..."
				MODEL=`ls -t ${EXP}/model/${NET_ID}/train_iter_*.caffemodel | head -n 1`
    fi
    echo Training2 net ${EXP}/${NET_ID}
    for pname in train solver2; do
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/${pname}.prototxt > ${CONFIG_DIR}/${pname}_${TRAIN_SET}.prototxt
    done
    CMD="${CAFFE_BIN} train \
         --solver=${CONFIG_DIR}/solver2_${TRAIN_SET}.prototxt \
         --gpu=${DEV_ID}"
    if [ $RESUME -le 0 ]; then
        CMD="${CMD} --weights=${MODEL}"
    fi
    if [ $RESUME -gt 0 ]; then
        CMD="${CMD} --snapshot=${SSTATE}"
    fi
		echo Running ${CMD} && ${CMD}
fi

## Test #2 on official test set
if [ ${RUN_TEST2} -eq 1 ]; then
    export GLOG_minloglevel=1
    for TEST_SET in test; do
        NUM_CLASSES=`cat ${EXP}/config/${NET_ID}/n_classes.txt`
        ((NUM_LABELS=NUM_CLASSES+1))  # Add background.
        TARGET_PERSON_SIZE=`cat ${EXP}/config/${NET_ID}/target_person_size.txt`
        TEST_SET=${TEST_SET}_${NUM_CLASSES}_${TARGET_PERSON_SIZE}_${TRAIN_SET_SUFFIX}
				TEST_ITER=`cat ${EXP}/list/${TEST_SET}.txt | wc -l`
				MODEL=${EXP}/model/${NET_ID}/test2.caffemodel
				if [ ! -f ${MODEL} ]; then
						MODEL=`ls -t ${EXP}/model/${NET_ID}/train2_iter_*.caffemodel | head -n 1`
				fi
        if [ ! -f ${MODEL} ]; then
            echo "Test model file not found: ${MODEL}!"
            exit 1
        fi
				echo Testing2 net ${EXP}/${NET_ID}
				FEATURE_DIR=${EXP}/features2/${NET_ID}
        mkdir -p ${FEATURE_DIR}/${TEST_SET}/seg_score
				sed "$(eval echo $(cat sub.sed))" \
						${CONFIG_DIR}/testpy.prototxt > ${CONFIG_DIR}/testpy_${TEST_SET}.prototxt
        CMD="./store_segmentation_results.py \
             ${CONFIG_DIR}/testpy_${TEST_SET}.prototxt \
             ${MODEL} \
             ${DATA_ROOT} \
             ${EXP}/list/${TEST_SET}.txt \
             ${FEATURE_DIR}/${TEST_SET}/seg_score \
             --caffe_install_path ${CAFFE_PYTHON} \
             --n_labels ${NUM_LABELS}"
        echo Running ${CMD} && ${CMD}
    done
fi

## Evaluation #2 specification (on val or test)
if [ ${RUN_EVALUATION2} -eq 1 ]; then
    export GLOG_minloglevel=1
    for TEST_SET in test; do
        echo Evaluating net ${EXP}/${NET_ID}
        FEATURE_DIR=${EXP}/features2/${NET_ID}
        NUM_CLASSES=`cat ${EXP}/config/${NET_ID}/n_classes.txt`
        ((NUM_LABELS=NUM_CLASSES+1))  # Add background.
        TARGET_PERSON_SIZE=`cat ${EXP}/config/${NET_ID}/target_person_size.txt`
        TEST_SET=${TEST_SET}_${NUM_CLASSES}_${TARGET_PERSON_SIZE}_${TRAIN_SET_SUFFIX}
        CMD="./evaluate_segmentation.py \
             ${EXP}/list/${TEST_SET}.txt \
             ${DATA_ROOT} \
             ${FEATURE_DIR}/${TEST_SET}/seg_score
             ${NUM_LABELS}"
        echo Running ${CMD} && ${CMD}
    done
fi
