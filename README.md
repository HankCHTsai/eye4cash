# eye4cash
<a id='Eval'></a>

This folder contains USD and NT coin(s) and banknote(s) image recognition including
its model training with TensorFlow slim framework and an example application utilizing
TensorFlow for Android devices.

## Description
<a id='Eval'></a>

model folder: [TensorFlow slim framework](https://github.com/tensorflow/models/tree/master/slim) to train cash recognition model in deep learning
technique. You can follow [slim README](model/slim/README.md) introduction to build the cash model 
for {NTD-1, NTD-5, NTD-10, NTD-50, NTD-100, NTD-500, NDT-1000, USD-1c, USD-10c, USD-25c,
USD-1, USD-20, USD-100} classes via my adding support with cash dataset components.

model/trained folder: You can find the latest trained model in this folder. It is fine-tune trained with MobileNet_v1 from
its [pre-trained model](http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz)

android folder: For TensorFlow android apk building referenced to [TensorFlow android demo example](https://github.com/tensorflow/tensorflow/tree/r1.2/tensorflow/examples/android)
You can follow [its README](android/README.md) to build android apk with TensorFlow support for inference with trained model.

## Current implement status
<a id='Eval'></a>

Want to quick success with courageous thinking: multiple classes regression to do multiple kinds of coins/banknotes counting,
not to use object detection which requires much effort on labeling.
Spend too much time to try unfamiliar interface on customizing label in tfrecord and loss function, so the counting and android
integration are not ready, now.

## Fine-tune train command
<a id='Eval'></a>

Download MobileNet_v1 [pre-trained model](http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz) and unzip it.
Then, go to model\slim folder and use the below command to do fine-tune train.
```shell
$ DATASET_DIR=/tmp/CashDataset
$ TRAIN_DIR=/tmp/cash-models/mobilenet_v1
$ CHECKPOINT_PATH=/tmp/pre-trianed/mobilenet_v1_1.0_224.ckpt
$ python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cash \
    --dataset_split_name=train \
    --model_name=mobilenet_v1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=MobilenetV1/Logits \
    --trainable_scopes=MobilenetV1/Logits
```

# Evaluating performance of a model
<a id='Eval'></a>

To evaluate the performance of a model (whether my trained or your own),
you can use the eval_image_classifier.py script, as shown below.

Below we give an example of downloading my trained mobilenet-v1 model and
evaluating it on your captured cash picture.

```shell
CHECKPOINT_FILE = ${CHECKPOINT_DIR}/cash_mobilenet_v1.ckpt  # Example
$ python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cash \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1
```

