FROM node:24-alpine AS build

WORKDIR /app
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/embed-react/package.json packages/embed-react/package.json
COPY packages/embed-element/package.json packages/embed-element/package.json
COPY apps/mock-logistics-partner/package.json apps/mock-logistics-partner/package.json
COPY apps/mock-logistics-partner-multitenant/package.json apps/mock-logistics-partner-multitenant/package.json
COPY apps/mock-logistics-partner-mcp/package.json apps/mock-logistics-partner-mcp/package.json
COPY apps/mock-logistics-partner-angular/package.json apps/mock-logistics-partner-angular/package.json
RUN pnpm install --frozen-lockfile
COPY packages/embed-react packages/embed-react
COPY packages/embed-element packages/embed-element
COPY apps/mock-logistics-partner apps/mock-logistics-partner
COPY apps/mock-logistics-partner-multitenant apps/mock-logistics-partner-multitenant
COPY apps/mock-logistics-partner-mcp apps/mock-logistics-partner-mcp
COPY apps/mock-logistics-partner-angular apps/mock-logistics-partner-angular
RUN pnpm build

FROM node:24-alpine

ENV NODE_ENV=production
WORKDIR /app
RUN corepack enable
COPY --from=build /app /app
EXPOSE 8787 8788 8789 8790
CMD ["pnpm", "--filter", "@autogpt/mock-logistics-partner", "start"]
