docker buildx build --platform linux/arm64 --load -t verekia/nanothree .
docker save -o /tmp/nanothree.tar verekia/nanothree
scp /tmp/nanothree.tar midgar:/tmp/
ssh midgar docker load --input /tmp/nanothree.tar
ssh midgar docker compose up -d nanothree