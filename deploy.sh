docker buildx build --platform linux/arm64 --load -t verekia/nanothree .
docker save verekia/nanothree | gzip > /tmp/nanothree.tar.gz
scp /tmp/nanothree.tar.gz midgar:/tmp/
ssh midgar docker load --input /tmp/nanothree.tar.gz
ssh midgar docker compose up -d nanothree