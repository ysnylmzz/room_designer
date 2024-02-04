
## Building Docker
'''
sudo docker build --build-arg ID=your_github_id --build-arg KEY=your_token_key  -t sd .
'''

## Run Dockerfile
'''
docker run --runtime=nvidia -v ${PWD}:/app  --shm-size="16g" -it sd 
'''