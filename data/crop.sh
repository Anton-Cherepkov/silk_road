python3 data/crop_pathes.py \
    --names data/data/non_cropped/train.txt \
    --src_dataset_folder data/data/non_cropped \
    --dst_dataset_folder data/data/cropped_train \
    --window_size 500 \
    --stride 500 && \
python3 data/crop_pathes.py \
    --names data/data/non_cropped/test.txt \
    --src_dataset_folder data/data/non_cropped \
    --dst_dataset_folder data/data/cropped_test \
    --window_size 500 \
    --stride 500