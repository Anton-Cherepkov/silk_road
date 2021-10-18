dataset structure:
```
dataset
  images
    xxx
  masks
    road
      xxx
    looks_like
      xxx
```

mask is a 1-channel uint8 png image


copy tif's to
```
data/data/non_cropped/images
```

run (only if you want new train/test split since it is fixed)
```
data/make_split.py
```

run (optional since masks already pushed)
```
data/draw_labels.py
```

run
```
data/crop.sh
```

run
```
data/make_split_for_crops.py
```
