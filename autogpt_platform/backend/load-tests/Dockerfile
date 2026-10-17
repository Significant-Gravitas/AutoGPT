# The k6 the load tests run on: the pinned grafana/k6 release plus xk6-sse,
# which k6 needs to read a Server-Sent Events stream as it arrives. Core k6
# buffers whole HTTP responses, so without it a copilot turn's time to first
# token cannot be measured from the client.
#
# k6.sh builds this image the first time it runs (about two minutes, needs
# network access for the Go modules) and reuses it afterwards.
ARG K6_VERSION=2.3.0
ARG XK6_SSE_VERSION=v0.2.0

FROM grafana/xk6:latest AS builder
ARG K6_VERSION
ARG XK6_SSE_VERSION
RUN xk6 build "v${K6_VERSION}" \
  --with "github.com/phymbert/xk6-sse@${XK6_SSE_VERSION}" \
  --output /tmp/k6

FROM grafana/k6:${K6_VERSION}
COPY --from=builder /tmp/k6 /usr/bin/k6
