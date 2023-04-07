import os
import redis

def connect_to_redis() -> redis.StrictRedis:
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', 6379))
    redis_password = os.environ.get('REDIS_PASSWORD', None)
    redis_conn = redis.StrictRedis(host=redis_host, port=redis_port, password=redis_password)
    redis_conn.ping()
    return redis_conn