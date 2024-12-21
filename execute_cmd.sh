#!/bin/bash
set -e
set -x
echo "Starting robot server..."
CYCLONEDDS_URI=file:///home/FlexivPy/cyclonedds_v0.xml robot_server -cm 3  --path /home/FlexivPy/FlexivPy/assets/ -rcf /home/FlexivPy/flexivpy_bridge/config.yaml
