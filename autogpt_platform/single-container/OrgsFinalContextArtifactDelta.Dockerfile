# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM node:24.18.0-bookworm-slim@sha256:6f7b03f7c2c8e2e784dcf9295400527b9b1270fd37b7e9a7285cf83b6951452d AS frontend

WORKDIR /app
RUN corepack enable
COPY autogpt_platform/frontend/package.json autogpt_platform/frontend/pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm pnpm install --frozen-lockfile
COPY autogpt_platform/frontend/ ./

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_AGPT_SERVER_URL=/_agpt/api \
    NEXT_PUBLIC_AGPT_WS_SERVER_URL=/_agpt/ws \
    NEXT_PUBLIC_FRONTEND_BASE_URL="" \
    NEXT_PUBLIC_APP_ENV=local \
    NEXT_PUBLIC_BEHAVE_AS=LOCAL \
    NEXT_PUBLIC_LAUNCHDARKLY_ENABLED=false \
    NEXT_PUBLIC_FORCE_FLAG_SHOW_ORG_SETTINGS=true \
    NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS=true \
    NEXT_PUBLIC_FORCE_FLAG_GRAPHITI_MEMORY=true \
    NEXT_PUBLIC_FORCE_FLAG_ARTIFACTS=true \
    NEXT_PUBLIC_FORCE_FLAG_ARTIFACTS_PAGE=true \
    NEXT_PUBLIC_FORCE_FLAG_CHAT_WORKSPACE_FILES=true \
    NEXT_PUBLIC_FORCE_FLAG_CHAT_SEARCH=true \
    NEXT_PUBLIC_FORCE_FLAG_CHAT_SHARING=true \
    NEXT_PUBLIC_SOURCEMAPS=false \
    NEXT_PUBLIC_TURNSTILE=disabled \
    NEXT_PUBLIC_VAPID_PUBLIC_KEY="" \
    BETTER_AUTH_SECRET=build-only-placeholder-not-used-at-runtime \
    DATABASE_URL="postgresql:///postgres?host=%2Frun%2Fpostgresql&user=autogpt_frontend"

RUN pnpm run generate:api \
    && NODE_OPTIONS="--max-old-space-size=8192" pnpm build \
    && rm -f .env .env.local .env.production

FROM autogpt-orgs-stack:orgs-final-copilot-context-20260828

ENV FORCE_FLAG_CHAT_SHARING=true

ARG VCS_REF=orgs-final-polish-20260828
ARG IMAGE_VERSION=orgs-final-polish-20260828

LABEL org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

COPY autogpt_platform/backend/backend/ /app/autogpt_platform/backend/backend/

RUN rm -rf /app/frontend
COPY --from=frontend /app/.next/standalone /app/frontend
COPY --from=frontend /app/.next/static /app/frontend/.next/static
COPY --from=frontend /app/public /app/frontend/public

RUN rm -rf \
        /app/frontend/node_modules/.pnpm/esbuild@* \
        /app/frontend/node_modules/.pnpm/@esbuild+linux-*@* \
    && find /app/frontend/node_modules -type l \
        \( -name esbuild -o -path '*/node_modules/@esbuild/linux-*' \) -delete \
    && rm -rf /app/frontend/.next/cache \
    && ln -s /data/cache/next /app/frontend/.next/cache
