
# Inference in docker

### Building Docker
```
cd docker
sudo docker build -t sd .
```

### Run App in Docker 
run docker
```
docker run --runtime=nvidia -v ${PWD}:/app  --shm-size="16g" -it sd 
```
run main script

```
cd app
python main.py
```

send request to api

```
cd app
python request.py
```

send concurent request 
```
cd app
python request_concurrent.py
```


# Inference without api and docker

install requirements
```
pip install -r requirements.py
```
run inference scrip  

```
python inference.py
```
