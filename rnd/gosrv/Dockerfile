# Build stage
FROM golang:1.23.1-alpine AS builder

WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies
RUN go mod download

# Copy the source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o main .

# Run stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates
ENV GIN_MODE=release

WORKDIR /root/

# Copy the pre-built binary file from the previous stage
COPY --from=builder /app/main .
COPY --from=builder /app/config.yaml .

# Expose port 8080 to the outside world
EXPOSE 8080

# Command to run the executable
CMD ["./main"]
