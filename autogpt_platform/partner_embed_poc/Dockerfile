FROM node:24-alpine AS build

WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/embed-react/package.json packages/embed-react/package.json
COPY apps/mock-forwarding-digital/package.json apps/mock-forwarding-digital/package.json
RUN pnpm install --frozen-lockfile
COPY packages/embed-react packages/embed-react
COPY apps/mock-forwarding-digital apps/mock-forwarding-digital
RUN pnpm build

FROM node:24-alpine

ENV NODE_ENV=production
WORKDIR /app
RUN corepack enable
COPY --from=build /app /app
EXPOSE 8787
CMD ["pnpm", "--filter", "@autogpt/mock-forwarding-digital", "start"]
