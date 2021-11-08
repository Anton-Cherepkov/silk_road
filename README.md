# Решение для распознавания и разметки пешеходных дорог
## Веб-интерфейс
### Системные требования:
1. Docker
2. NVIDIA CUDA 11.1
3. \>= 1 NVIDIA GPU с объемом видеопамяти не менее 4 Гб
4. OC: Linux (рекомендуемо), Windows, macOS
### Инструкция по запуску:
1. Склонируйте или скачайте текущий репозиторий.

2. Для сборки docker-контейнера перейдите в корневую папку проекта и выполните:

`docker build . -t silk_road`

3. Для запуска веб интерфейса выполните команду:

`docker run --gpus all -d -p 8011:8011 silk_road`

Если при запуске появляется ошибка, значит, вероятнее всего, Ваша версия Docker не поддерживает запуск контейнера с GPU. Попробуйте запустить со следующей командой:

`docker run -d -p 8011:8011 silk_road`

### Инструкция по остановке:
1. Для остановки веб-интерфейса выполните команду:

`docker stop [CONTAINER ID]`

### Инструкция по работе
0. Запустите веб-интерфейс по инструкции. Подождите некоторое время после выполнения пункта 3.
1. Перейдите по адресу http://localhost:8011/.
2. В соответствующие поля загрузите TIF-изображение размера 5000x5000 и tfw-файл (опционально).
3. Нажмите обработать. Подождите некоторое время.
4. Скачайте необходимые файлы на странице с результатами.

### Возможности:
* Загрузка нескольких файлов одновременно;
* Просмотр распознанных дорожек с наложением на карту;
* Скачивание выходных файлов:
* * Объединенный Shapefile для всех загруженных изображений;
* * Shapefile для каждого изображения;
* * Визуализация дорожек на каждом изображении;
* * Визуализация дорожек после постпроцессинга на каждом изображении;
* * Маска вероятностей дорожек для каждого изображения.
* Развертывание системы на машине без GPU. В таком случае вычисления производятся на CPU.

### Troubleshooting
Если по каким-то причинам веб-интерфейс не работает после выполнения пункта 3 из инструкции по запуску, то возникшие ошибки можно изучить с помощью команды:

`docker logs [CONTAINER ID]`

### Скачать уже собранный образ из Docker Hub
Так как образ находится в закрытом доступе, то сперва необходимо авторизоваться:

`docker login -u acherepkov`

Система должна запросить пароль:

`bdb82d7c-08ef-4a5e-8326-ff3d8f70090b`

Скачать собранный образ можно следующей командой:

`docker pull acherepkov/silk_road`

Для дальнейшего запуска и разворачивания веб-интерфейса следуйте инструкции по запуску, начиная с шага 3.

### Инструкция по контрольной компиляции
*Внимание, это черновик! Вскоре он подлежит перепроверке и, возможно, обновлению.*

1. Выполните установку Ubuntu 18.04.6 AMD64. Скачать дистрибутив данной ОС можно тут: 

https://releases.ubuntu.com/18.04/ubuntu-18.04.6-desktop-amd64.iso

Инструкция по установке доступна, например, тут: https://losst.ru/ustanovka-ubuntu-18-04

2. Установите драйвера CUDA 11.1
* Выполните команду

`sudo apt-get -y install gcc make`

* Выполните команду

`wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run`

* Выполните команду

`sudo sh cuda_11.1.0_455.23.05_linux.run`

* На данном этапе может появиться сообщение о том, что установка не удалась. В таком случае перезагрузите компьютер и снова выполните установку с помощью команды:

`sudo sh cuda_11.1.0_455.23.05_linux.run`

* Перезагрузите компьютер
* Проверьте успешность установки драйверов с помощью команды `nvidia-smi`. Должен отобразиться список видеокарт, а также версия CUDA в верхнем правом углу.

3. Установите Docker
* Выполните поочередно следующие команды
```
sudo apt-get update
```
```
sudo apt-get -y install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```
```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```
```
sudo apt-get update
```
```
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
```
* Перезагрузите компьютер
* Чтобы убедиться, что установка Docker прошла успешно, выполните
```
sudo docker run hello-world
```
Выполните следующие команды:
```
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
4. Склонируйте или скачайте текущий репозиторий

5. Для сборки docker-контейнера перейдите в корневую папку проекта и выполните:

`sudo docker build . -t silk_road`

6. Для запуска веб интерфейса выполните команду:

`sudo docker run --gpus all -d -p 8011:8011 silk_road`
