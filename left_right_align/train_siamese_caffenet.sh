#!/usr/bin/env sh

TOOLS=~/caffe/build/tools

$TOOLS/caffe train --solver=siamese_caffenet_solver.prototxt
