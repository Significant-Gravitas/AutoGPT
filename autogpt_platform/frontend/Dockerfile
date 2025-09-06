# Base stage for both dev and prod
FROM node:21-alpine AS base
WORKDIR /app
RUN corepack enable
COPY autogpt_platform/frontend/package.json autogpt_platform/frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm pnpm install --frozen-lockfile

# Build stage for prod
FROM base AS build

COPY autogpt_platform/frontend/ .
# Allow CI to opt-in to Playwright test build-time flags
ARG NEXT_PUBLIC_PW_TEST="false"
ENV NEXT_PUBLIC_PW_TEST=$NEXT_PUBLIC_PW_TEST
ENV NODE_ENV="production"
# Merge env files appropriately based on environment
RUN if [ -f .env.production ]; then \
      # In CI/CD: merge defaults with production (production takes precedence)
      cat .env.default .env.production > .env.merged && mv .env.merged .env.production; \
    elif [ -f .env ]; then \
      # Local with custom .env: merge defaults with .env
      cat .env.default .env > .env.merged && mv .env.merged .env; \
    else \
      # Local without custom .env: use defaults
      cp .env.default .env; \
    fi
RUN pnpm run generate:api
# In CI, we want NEXT_PUBLIC_PW_TEST=true during build so Next.js inlines it
RUN if [ "$NEXT_PUBLIC_PW_TEST" = "true" ]; then NEXT_PUBLIC_PW_TEST=true NODE_OPTIONS="--max-old-space-size=4096" pnpm build; else NODE_OPTIONS="--max-old-space-size=4096" pnpm build; fi

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
