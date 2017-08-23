#!/usr/bin/sh --login

if [ ! -f "deeplab_resnet.ckpt" ]; then
	cat model-trainsets.a* | tar -jxv
fi

if [ ! -f "/home/VOCdevkit/train.txt" ]; then
	unzip voc2012-500x500.zip -d /home/VOCdevkit
fi

echo "Usage: $0 [step]"
echo "    sample: $0 8000"

ARGS=
if [ -f "snapshots/model.ckpt-$1.index" ]; then
	ARGS="--restore-from snapshots/model.ckpt-$1"
fi

ARGS=
if [ -f "snapshots_finetune/model.ckpt-$1.index" ]; then
    ARGS="--restore-from snapshots_finetune/model.ckpt-$1"
fi

nohup python evaluate.py --data-list /home/VOCdevkit/train.txt $ARGS > evaluate.log 2>&1 &
pidstat -C python -r 1

