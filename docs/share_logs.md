## Share your logs with us to help improve Auto-GPT

Do you notice weird behavior with your agent? Do you have an interesting use case? Do you have a bug you want to report? 
Follow the steps below to enable your logs and upload them. You can include these logs when making an issue report or discussing an issue with us.

### Enable Debug Logs
Activity and error logs are located in the `./output/logs`

To print out debug logs:

``` shell
./run.sh --debug     # on Linux / macOS

.\run.bat --debug    # on Windows

docker-compose run --rm auto-gpt --debug    # in Docker
```
