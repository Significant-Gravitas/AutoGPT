# Base stage for both dev and prod
FROM node:21-alpine AS base
WORKDIR /app
COPY autogpt_builder/package.json autogpt_builder/yarn.lock ./
RUN yarn install --frozen-lockfile

# Dev stage
FROM base AS dev
ENV NODE_ENV=development
COPY autogpt_builder/ .
EXPOSE 3000
CMD ["npm", "run", "dev"]

# Build stage for prod
FROM base AS build
COPY autogpt_builder/ .
RUN npm run build

# Prod stage
FROM node:21-alpine AS prod
ENV NODE_ENV=production
WORKDIR /app

COPY --from=build /app/package.json /app/yarn.lock ./
RUN yarn install --frozen-lockfile

COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/next.config.mjs ./next.config.mjs

EXPOSE 3000
CMD ["npm", "start"]
