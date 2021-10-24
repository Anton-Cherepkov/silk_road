# SilkRoad

## Установка и запуск docker-контейнера
Для запуска всех нижеследующих команд необходимо запустить docker-контейнер:
```
git clone https://github.com/Anton-Cherepkov/silk_road.git
cd silk_road
./make.sh build
./make.sh run
./make.sh lfs_pull
```
## Запуск сервера для web-интерфейса

После запуска нижеследующей команды станет доступен web-интерфейс по ссылке [localhost:8011](localhost:8011)
```
./make.sh run_web_ui
```

## Интерфейс командной строки для предсказания шейпфайлов на папке со снимками 

Если имеется исходная папка `ПУТЬ_ДО_ПАПКИ`, содержащая в себе для каждого снимка `IMG.tif`, также файл `IMG.tfw` с координатами `EPSG:32637`, можно используя интерфейс командной строки, сгенерировать shapefile'ы:

```
./make.sh run_inference ПУТЬ_ДО_ПАПКИ ПУТЬ_ДО_ПАПКИ_С_СЫРЫМИ_МАСКАМИ_И_ИХ_ВИЗУАЛИЗАЦИЕЙ ПУТЬ_ДО_ПАПКИ_В_КОТОРОЙ_БУДУТ_ЛЕЖАТЬ_SHAPE_ФАЙЛЫ
```
Пути до папок не должны заканчиваться `/`.
Все папки должны быть подпапками корневой папки проекта (silk_road) и пути должны быть относительно неё. 
То есть, если есть папка с изображениями `/home/abc/silk_road/datasets/images` и нужно, чтобы сырые маски и их визуализация оказались в папках `/home/abs/silk_road/predicts/images_masks`, а shape-файлы в папке `/home/abs/silk_road/predicts/images_shapefiles`, то команда будет выглядеть следующим образом:
```
./make.sh run_inference datasets/images predicts/images_masks predicts/images_shapefiles
```
