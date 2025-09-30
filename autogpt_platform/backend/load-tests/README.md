# AutoGPT Platform Load Tests

Clean, streamlined load testing infrastructure for the AutoGPT Platform using k6.

## 🚀 Quick Start

```bash
# 1. Set up Supabase service key (required for token generation)
export SUPABASE_SERVICE_KEY="your-supabase-service-key"

# 2. Generate pre-authenticated tokens (first time setup - creates 160+ tokens with 24-hour expiry)  
node generate-tokens.js --count=160

# 3. Set up k6 cloud credentials (for cloud testing - see Credential Setup section below)
export K6_CLOUD_TOKEN="your-k6-cloud-token"  
export K6_CLOUD_PROJECT_ID="4254406"

# 4. Run orchestrated load tests locally
node orchestrator/orchestrator.js DEV local

# 5. Run orchestrated load tests in k6 cloud (recommended)
node orchestrator/orchestrator.js DEV cloud
```

## 📋 Load Test Orchestrator

The AutoGPT Platform uses a comprehensive load test orchestrator (`orchestrator/orchestrator.js`) that runs 12 optimized tests with maximum VU counts:

### Available Tests

#### Basic Tests (Simple validation)

- **connectivity-test**: Basic connectivity and authentication validation
- **single-endpoint-test**: Individual API endpoint testing with high concurrency

#### API Tests (Core functionality)

- **core-api-test**: Core API endpoints (`/api/credits`, `/api/graphs`, `/api/blocks`, `/api/executions`)
- **graph-execution-test**: Complete graph creation and execution pipeline

#### Marketplace Tests (User-facing features)

- **marketplace-public-test**: Public marketplace browsing and search
- **marketplace-library-test**: Authenticated marketplace and user library operations

#### Comprehensive Tests (End-to-end scenarios)

- **comprehensive-test**: Complete user journey simulation with multiple operations

### Test Modes

- **Local Mode**: 5 VUs × 30s - Quick validation and debugging
- **Cloud Mode**: 80-160 VUs × 3-6m - Real performance testing

## 🛠️ Usage

### Basic Commands

```bash
# Run 12 optimized tests locally (for debugging)
node orchestrator/orchestrator.js DEV local

# Run 12 optimized tests in k6 cloud (recommended for performance testing)
node orchestrator/orchestrator.js DEV cloud

# Run against production (coordinate with team!)
node orchestrator/orchestrator.js PROD cloud

# Run individual test directly with k6
K6_ENVIRONMENT=DEV VUS=100 DURATION=3m k6 run tests/api/core-api-test.js
```

### NPM Scripts

```bash
# Run orchestrator locally
npm run local

# Run orchestrator in k6 cloud
npm run cloud
```

## 🔧 Test Configuration

### Pre-Authenticated Tokens

- **Generation**: Run `node generate-tokens.js --count=160` to create tokens
- **File**: `configs/pre-authenticated-tokens.js` (gitignored for security)  
- **Capacity**: 160+ tokens supporting high-concurrency testing
- **Expiry**: 24 hours (86400 seconds) - extended for long-duration testing
- **Benefit**: Eliminates Supabase auth rate limiting at scale
- **Regeneration**: Run `node generate-tokens.js --count=160` when tokens expire after 24 hours

### Environment Configuration

- **LOCAL**: `http://localhost:8006` (local development)
- **DEV**: `https://dev-api.agpt.co` (development environment)
- **PROD**: `https://api.agpt.co` (production environment - coordinate with team!)

## 📊 Performance Testing Features

### Real-Time Monitoring

- **k6 Cloud Dashboard**: Live performance metrics during cloud test execution
- **URL Tracking**: Test URLs automatically saved to `k6-cloud-results.txt`
- **Error Tracking**: Detailed failure analysis and HTTP status monitoring
- **Custom Metrics**: Request success/failure rates, response times, user journey tracking
- **Authentication Monitoring**: Tracks auth success/failure rates separately from HTTP errors

### Load Testing Capabilities

- **High Concurrency**: Up to 160+ virtual users per test
- **Authentication Scaling**: Pre-auth tokens support 160+ concurrent users
- **Sequential Execution**: Multiple tests run one after another with proper delays
- **Cloud Infrastructure**: Tests run on k6 cloud servers for consistent results
- **ES Module Support**: Full ES module compatibility with modern JavaScript features

## 📈 Performance Expectations

### Validated Performance Limits

- **Core API**: 100+ VUs successfully handling `/api/credits`, `/api/graphs`, `/api/blocks`, `/api/executions`
- **Graph Execution**: 80+ VUs for complete workflow pipeline  
- **Marketplace Browsing**: 160 VUs for public marketplace access (verified)
- **Marketplace Library**: 160 VUs for authenticated library operations (verified)
- **Authentication**: 160+ concurrent users with pre-authenticated tokens

### Target Metrics

- **P95 Latency**: Target < 5 seconds (marketplace), < 2 seconds (core API)
- **P99 Latency**: Target < 10 seconds (marketplace), < 5 seconds (core API)
- **Success Rate**: Target > 95% under normal load
- **Error Rate**: Target < 5% for all endpoints

### Recent Performance Results (160 VU Test - Verified)

- **Marketplace Library Operations**: 500-1000ms response times at 160 VUs
- **Authentication**: 100% success rate with pre-authenticated tokens
- **Library Journeys**: 5 operations per journey completing successfully
- **Test Duration**: 6+ minutes sustained load without degradation
- **k6 Cloud Execution**: Stable performance on Amazon US Columbus infrastructure

## 🔍 Troubleshooting

### Common Issues

**1. Authentication Failures**

```
❌ No valid authentication token available
❌ Token has expired
```

- **Solution**: Run `node generate-tokens.js --count=160` to create fresh 24-hour tokens
- **Note**: Use `--count` parameter to generate appropriate number of tokens for your test scale

**2. Cloud Credentials Missing**

```
❌ Missing k6 cloud credentials
```

- **Solution**: Set `K6_CLOUD_TOKEN` and `K6_CLOUD_PROJECT_ID=4254406`

**3. k6 Cloud VU Scaling Issue**

```
❌ Test shows only 5 VUs instead of requested 100+ VUs
```

- **Problem**: Using `K6_ENVIRONMENT=DEV VUS=160 k6 cloud run test.js` (incorrect)
- **Solution**: Use `k6 cloud run --env K6_ENVIRONMENT=DEV --env VUS=160 test.js` (correct)
- **Note**: The unified test runner (`run-tests.js`) already uses the correct syntax

**4. Setup Verification Failed**

```
❌ Verification failed
```

- **Solution**: Check tokens exist and local API is accessible

### Required Setup

**1. Supabase Service Key (Required for all testing):**

```bash
# Option 1: From your local environment (if available)
export SUPABASE_SERVICE_KEY="your-supabase-service-key"

# Option 2: From Kubernetes secret (for platform developers)
kubectl get secret supabase-service-key -o jsonpath='{.data.service-key}' | base64 -d

# Option 3: From Supabase dashboard
# Go to Project Settings > API > service_role key (never commit this!)
```

**2. Generate Pre-Authenticated Tokens (Required):**

```bash
# Creates 160 tokens with 24-hour expiry - prevents auth rate limiting
node generate-tokens.js --count=160

# Generate fewer tokens for smaller tests (minimum 10)
node generate-tokens.js --count=50

# Regenerate when tokens expire (every 24 hours)
node generate-tokens.js --count=160
```

**3. k6 Cloud Credentials (Required for cloud testing):**

```bash
# Get from k6 cloud dashboard: https://app.k6.io/account/api-token
export K6_CLOUD_TOKEN="your-k6-cloud-token"
export K6_CLOUD_PROJECT_ID="4254406"  # AutoGPT Platform project ID

# Verify credentials work by running orchestrator
node orchestrator/orchestrator.js DEV cloud
```

## 📂 File Structure

```
load-tests/
├── README.md                              # This documentation
├── generate-tokens.js                     # Generate pre-auth tokens (MAIN TOKEN SETUP)
├── package.json                           # Node.js dependencies and scripts
├── orchestrator/
│   └── orchestrator.js                    # Main test orchestrator (MAIN ENTRY POINT)
├── configs/
│   ├── environment.js                     # Environment URLs and configuration
│   └── pre-authenticated-tokens.js        # Generated tokens (gitignored)
├── tests/
│   ├── basic/
│   │   ├── connectivity-test.js           # Basic connectivity validation
│   │   └── single-endpoint-test.js        # Individual API endpoint testing
│   ├── api/
│   │   ├── core-api-test.js               # Core authenticated API endpoints
│   │   └── graph-execution-test.js        # Graph workflow pipeline testing
│   ├── marketplace/
│   │   ├── public-access-test.js          # Public marketplace browsing
│   │   └── library-access-test.js         # Authenticated marketplace/library
│   └── comprehensive/
│       └── platform-journey-test.js       # Complete user journey simulation
├── results/                               # Local test results (auto-created)
├── unified-results-*.json                 # Orchestrator results (auto-created)
└── *.log                                  # Test execution logs (auto-created)
```

## 🎯 Best Practices

1. **Generate Tokens First**: Always run `node generate-tokens.js --count=160` before testing
2. **Local for Development**: Use `DEV local` for debugging and development
3. **Cloud for Performance**: Use `DEV cloud` for actual performance testing
4. **Monitor Real-Time**: Check k6 cloud dashboards during test execution
5. **Regenerate Tokens**: Refresh tokens every 24 hours when they expire
6. **Unified Testing**: Orchestrator runs 12 optimized tests automatically

## 🚀 Advanced Usage

### Direct k6 Execution

For granular control over individual test scripts:

```bash
# k6 Cloud execution (recommended for performance testing)
# IMPORTANT: Use --env syntax for k6 cloud to ensure proper VU scaling
k6 cloud run --env K6_ENVIRONMENT=DEV --env VUS=160 --env DURATION=5m --env RAMP_UP=30s --env RAMP_DOWN=30s tests/marketplace/library-access-test.js

# Local execution with cloud output (debugging)
K6_ENVIRONMENT=DEV VUS=10 DURATION=1m \
k6 run tests/api/core-api-test.js --out cloud

# Local execution with JSON output (offline testing)
K6_ENVIRONMENT=DEV VUS=10 DURATION=1m \
k6 run tests/api/core-api-test.js --out json=results.json
```

### Custom Token Generation

```bash
# Generate specific number of tokens
node generate-tokens.js --count=200

# Generate tokens with custom timeout
node generate-tokens.js --count=100 --timeout=60
```

## 🔗 Related Documentation

- [k6 Documentation](https://k6.io/docs/)
- [AutoGPT Platform API Documentation](https://docs.agpt.co/)
- [k6 Cloud Dashboard](https://significantgravitas.grafana.net/a/k6-app/)

For questions or issues, please refer to the [AutoGPT Platform issues](https://github.com/Significant-Gravitas/AutoGPT/issues).
