import { Ratelimit } from "@upstash/ratelimit"; // for deno: see above
import { Redis } from "@upstash/redis";
import { env } from "../env/server.mjs";

const redis = new Redis({
  url: env.UPSTASH_REDIS_REST_URL ?? "",
  token: env.UPSTASH_REDIS_REST_TOKEN ?? "",
});

export const rateLimiter = new Ratelimit({
  redis: redis,
  limiter: Ratelimit.slidingWindow(
    env.RATE_LIMITER_REQUESTS_PER_MINUTE ?? 100,
    "60 s"
  ),
  analytics: true,
  prefix: "@upstash/ratelimit",
});
