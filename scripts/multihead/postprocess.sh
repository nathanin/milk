#!/usr/bin/env bash

ls /mnt/linux-data/va-pnbx/*.att.10x.shallow.npy > attention-imgs.txt
ls /mnt/linux-data/va-pnbx/*.img.jpg > slide-imgs.txt

python Overlay_attention.py slide-imgs.txt attention-imgs.txt --color cubehelix

cp -v /mnt/linux-data/va-pnbx/*.attcol.jpg ./shallow-10x-10heads/result_test_svs/