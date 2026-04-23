#!/usr/bin/env bash
set -e

bash train_base.sh
bash train_expo.sh
bash train_fusion.sh
