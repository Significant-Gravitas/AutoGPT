# Base stage for both dev and prod
FROM node:21-alpine AS base
WORKDIR /app
RUN corepack enable
COPY autogpt_platform/frontend/package.json autogpt_platform/frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm pnpm install --frozen-lockfile

# Build stage for prod
FROM base AS build

COPY autogpt_platform/frontend/ .
RUN if [ -f .env ]; then \
      cat .env.default .env > .env.merged && mv .env.merged .env; \
    else \
      cp .env.default .env; \
    fi
RUN pnpm run generate:api
RUN pnpm build

# Prod stage - based on NextJS reference Dockerfile https://github.com/vercel/next.js/blob/64271354533ed16da51be5dce85f0dbd15f17517/examples/with-docker/Dockerfile
FROM node:21-alpine AS prod
ENV NODE_ENV=production
ENV HOSTNAME=0.0.0.0
WORKDIR /app

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

RUN mkdir .next
RUN chown nextjs:nodejs .next

COPY --from=build --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=build --chown=nextjs:nodejs /app/.next/static ./.next/static

COPY --from=build /app/public ./public

USER nextjs

EXPOSE 3000
CMD ["node", "server.js"]
