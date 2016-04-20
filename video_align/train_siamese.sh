#!/usr/bin/env sh

CAFFE_PATH=~/caffe

TOOLS=$CAFFE_PATH/build/tools

$TOOLS/caffe train --solver=./siamese_solver.prototext
