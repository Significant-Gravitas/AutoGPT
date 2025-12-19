# Build stage
FROM rust:1.75-slim as builder

RUN apt-get update && \
    apt-get install -y pkg-config libssl-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY src ./src

# Build release binary with size optimizations
RUN cargo build --release && \
    strip /app/target/release/db-migrate && \
    rm -rf /app/target/release/deps /app/target/release/build /app/target/release/.fingerprint /app/target/release/incremental

# Runtime stage - minimal image (~50MB)
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libpq5 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/*

# Copy only the binary
COPY --from=builder /app/target/release/db-migrate /usr/local/bin/db-migrate

# Run as non-root
RUN useradd -r -s /bin/false migrate
USER migrate

ENTRYPOINT ["db-migrate"]
