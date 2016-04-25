#!/usr/bin/env sh

TOOLS=~/caffe/python

$TOOLS/caffe train --solver=siamese_solver.prototxt
