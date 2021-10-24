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
Важно, чтобы до момента запуска скрипта папка `ПУТЬ_ДО_ПАПКИ_В_КОТОРОЙ_БУДУТ_ЛЕЖАТЬ_РЕЗУЛЬТЫ` не существовала; если в путях до папок есть пропелы, то их (пути) нужно взять в кавычки.