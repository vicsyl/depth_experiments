from pathlib import Path

to_split = """multi_view_training_dslr_undistorted/courtyard/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/pipes/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/delivery_area/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/playground/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/electro/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/relief/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/facade/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/relief_2/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/kicker/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/terrace/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/meadow/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/terrains/images/dslr_images_undistorted
multi_view_training_dslr_undistorted/office/images/dslr_images_undistorted"""

files = to_split.split("\n")
print(files)
print()

# MoGe
for f in files:
    key = Path(f).parent.parent.name
    # print(key)
    print(f"python infer.py --input ../datasets/eth_3d/multi_view_training_dslr_undistorted/{key}/images/dslr_images_undistorted --output out/eth_3d/{key} --maps")


# ml-depth-pro
# depth-pro-run -i ../datasets/phototourism/imw-2020-test/british_museum/ -o out/
for f in files:
    key = Path(f).parent.parent.name
    # print(key)
    print(f"depth-pro-run -i ../datasets/eth_3d/multi_view_training_dslr_undistorted/{key}/images/dslr_images_undistorted -o out/eth_3d/{key}")

# metric3D (v2)
# python hubconf.py --in_dir {IN_DIR} --out_dir {OUT_DIR}
for f in files:
    key = Path(f).parent.parent.name
    # print(key)
    log_file = "{log_file}"
    print(f"python hubconf.py --in_dir ../datasets/eth_3d/multi_view_training_dslr_undistorted/{key}/images/dslr_images_undistorted --out_dir out/eth_3d/{key}  2>&1 | tee ${log_file}")
