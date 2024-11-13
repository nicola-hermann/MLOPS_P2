docker build -t mlopsp2 .

docker run --env-file .env mlopsp2

docker rmi -f mlopsp2