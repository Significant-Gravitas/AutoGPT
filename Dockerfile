# Stage 1: Build environment
FROM python:3.9 as build-env
WORKDIR /app
COPY ./run .
COPY . .
RUN ./run


# Stage 2: Runtime environment
FROM python:3.9-slim as runtime-env
COPY --from=build-env /app /app
CMD ["./run"]