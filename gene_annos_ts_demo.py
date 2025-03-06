import glob
import os
import os.path as osp
from pathlib import Path

import numpy as np
import cv2
import json


"""
Generates annotation json file for twelve scenes and also outputs a shell script code to run inference
on this annotation file.
"""

def get_k(path):
    with open(os.path.join(path, "info.txt")) as fd:
        for l in fd.readlines():
            kv = l.split("=")
            if kv[0].strip() == "m_calibrationColorIntrinsic":
                l = [float(n) for n in kv[1].strip().split(" ")]
                K = np.array(l).reshape(4, 4)[:3, :3]
                return K

    assert False, "intrinsics not found"


def run_room(files, apt, room, small_data, max_count, add_intrinsics):

    code_root = "../"
    room_root = osp.join(code_root, f"data/twelve_scenes/{apt}_{room}")

    K = get_k(room_root)

    data_subdir = "small_data" if small_data else "data"
    rgb_root = osp.join(room_root, data_subdir)
    # depth_root = osp.join(room_root, 'depth')

    l2 = os.listdir(rgb_root)
    l = list(glob.glob(f"{rgb_root}/*.color.jpg"))
    l = sorted([Path(f).name for f in l])
    if max_count:
        l = l[:max_count]

    for rgb_file in l:

        rgb_path = osp.join(rgb_root, rgb_file).split(code_root)[-1]
        depth_scale = 256.

        meta_data = {}
        if add_intrinsics:
            cam_in = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
            meta_data['cam_in'] = cam_in
        meta_data['rgb'] = rgb_path
        # meta_data['depth'] = depth_path
        meta_data['depth_scale'] = depth_scale
        files.append(meta_data)


def template(fn):

    fn_log = fn.replace(".", "_")

    print(f"""#!/bin/bash

log_file=log_ts_{fn_log}.txt

python mono/tools/test_scale_cano.py \\
    'mono/configs/HourglassDecoder/vit.raft5.giant2.py' \\
    --load-from ./weight/metric_depth_vit_giant2_800k.pth \\
    --test_data_path ./data/twelve_scenes/annotations/{fn} \\
    --launcher None > ${{log_file}}  2>&1 &

echo "tail..."
sleep 1

tail -f ${{log_file}}
    """)


def run():

    code_root = "../"

    #for max_count in [None, 100]:
    for max_count in [100]:
        # FIXME deprecated
        small_data = False
        add_intrinsics = False
        apts_rooms = [('apt1', 'kitchen'), ('apt1', 'living'), ('apt2', 'bed'),
                      ('apt2', 'kitchen'), ('apt2', 'living'), ('apt2', 'luke'), ('office2', '5a'), ('office2', '5b')]

        # apts_rooms = [('apt1', 'kitchen'), ('apt1', 'living')]

        files = []
        for (apt, room) in sorted(apts_rooms):
            run_room(files, apt, room, small_data=small_data, max_count=max_count, add_intrinsics=add_intrinsics)

        files_dict = dict(files=files)
        small_sfx = f"_{max_count}" if max_count else "_all"
        small_sfx += "_K" if add_intrinsics else "_noK"
        fn = f"test_annotations{small_sfx}.json"
        with open(osp.join(code_root, f'./data/twelve_scenes/annotations/{fn}'), 'w') as f:
            json.dump(files_dict, f, indent=4)
        template(fn)


if __name__=='__main__':
    run()
