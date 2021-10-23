dst_folder="inference/images"

while read name; do
    scp "cl-dl04:/mnt/ssd/wrk/shubin/02.\ Решение\ для\ распознавания\ и\ разметки\ пешеходных\ маршрутов/Датасет_02/Датасет/${name}.tif" "${dst_folder}"
done < data/data/non_cropped/test.txt
