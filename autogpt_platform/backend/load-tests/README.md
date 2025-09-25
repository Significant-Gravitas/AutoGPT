# AutoGPT Platform Load Tests

Clean, streamlined load testing infrastructure for the AutoGPT Platform using k6.

## 🚀 Quick Start

```bash
# 1. Set up Supabase service key (required for token generation)
export SUPABASE_SERVICE_KEY="your-supabase-service-key"

# 2. Generate pre-authenticated tokens (first time setup - creates 150+ tokens with 24-hour expiry)
node generate-tokens.js

# 3. Set up k6 cloud credentials (for cloud testing)
export K6_CLOUD_TOKEN="your-k6-cloud-token"
export K6_CLOUD_PROJECT_ID="4254406"

# 4. Verify setup and run quick test
node run-tests.js verify

# 5. Run tests locally (development/debugging)
node run-tests.js run all DEV

# 6. Run tests in k6 cloud (performance testing)
node run-tests.js cloud all DEV
```

## 📋 Unified Test Runner

The AutoGPT Platform uses a single unified test runner (`run-tests.js`) for both local and cloud execution:

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
- **Cloud Mode**: 80-150 VUs × 3-5m - Real performance testing

## 🛠️ Usage

### Basic Commands

```bash
# List available tests and show cloud credentials status
node run-tests.js list

# Quick setup verification
node run-tests.js verify

# Run specific test locally  
node run-tests.js run core-api-test DEV

# Run multiple tests sequentially (comma-separated)
node run-tests.js run connectivity-test,core-api-test,marketplace-public-test DEV

# Run all tests locally
node run-tests.js run all DEV

# Run specific test in k6 cloud
node run-tests.js cloud core-api-test DEV

# Run all tests in k6 cloud
node run-tests.js cloud all DEV
```

### NPM Scripts

```bash
# Quick verification
npm run verify

# Run all tests locally
npm test  

# Run all tests in k6 cloud
npm run cloud
```

## 🔧 Test Configuration

### Pre-Authenticated Tokens
- **Generation**: Run `node generate-tokens.js` to create tokens
- **File**: `configs/pre-authenticated-tokens.js` (gitignored for security)
- **Capacity**: 150+ tokens supporting high-concurrency testing
- **Expiry**: 24 hours (86400 seconds) - extended for long-duration testing
- **Benefit**: Eliminates Supabase auth rate limiting at scale
- **Regeneration**: Run `node generate-tokens.js` when tokens expire after 24 hours

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

### Load Testing Capabilities
- **High Concurrency**: Up to 150+ virtual users per test
- **Authentication Scaling**: Pre-auth tokens support 150+ concurrent users
- **Sequential Execution**: Multiple tests run one after another with proper delays
- **Cloud Infrastructure**: Tests run on k6 cloud servers for consistent results

## 📈 Performance Expectations

### Validated Performance Limits
- **Core API**: 100 VUs successfully handling `/api/credits`, `/api/graphs`, `/api/blocks`, `/api/executions`
- **Graph Execution**: 80 VUs for complete workflow pipeline
- **Marketplace Browsing**: 150 VUs for public marketplace access
- **Authentication**: 150+ concurrent users with pre-authenticated tokens

### Target Metrics  
- **P95 Latency**: Target < 5 seconds (marketplace), < 2 seconds (core API)
- **P99 Latency**: Target < 10 seconds (marketplace), < 5 seconds (core API)
- **Success Rate**: Target > 95% under normal load
- **Error Rate**: Target < 5% for all endpoints

## 🔍 Troubleshooting

### Common Issues

**1. Authentication Failures**
```
❌ No valid authentication token available
```
- **Solution**: Run `node generate-tokens.js` to create fresh tokens

**2. Cloud Credentials Missing**  
```
❌ Missing k6 cloud credentials
```
- **Solution**: Set `K6_CLOUD_TOKEN` and `K6_CLOUD_PROJECT_ID=4254406`

**3. Setup Verification Failed**
```
❌ Verification failed
```
- **Solution**: Check tokens exist and local API is accessible

### Required Setup

**1. Supabase Service Key (Required for all testing):**
```bash
# Get service key from environment or Kubernetes
export SUPABASE_SERVICE_KEY="your-supabase-service-key"
```

**2. Generate Pre-Authenticated Tokens (Required):**
```bash
# Creates tokens with 24-hour expiry - prevents auth rate limiting  
node generate-tokens.js

# Regenerate when tokens expire
node generate-tokens.js
```

**3. k6 Cloud Credentials (Required for cloud testing):**
```bash
export K6_CLOUD_TOKEN="your-k6-cloud-token" 
export K6_CLOUD_PROJECT_ID="4254406"  # AutoGPT Platform project ID
```

## 📂 File Structure

```
load-tests/
├── README.md                              # This documentation
├── run-tests.js                           # Unified test runner (MAIN ENTRY POINT)
├── generate-tokens.js                     # Generate pre-auth tokens
├── package.json                           # Node.js dependencies and scripts
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
├── orchestrator/
│   └── comprehensive-orchestrator.js      # Full 25-test orchestration suite
├── results/                               # Local test results (auto-created)
├── k6-cloud-results.txt                   # Cloud test URLs (auto-created)
└── *.json                                 # Test output files (auto-created)
```

## 🎯 Best Practices

1. **Start with Verification**: Always run `node run-tests.js verify` first
2. **Local for Development**: Use `run` command for debugging and development
3. **Cloud for Performance**: Use `cloud` command for actual performance testing
4. **Monitor Real-Time**: Check k6 cloud dashboards during test execution
5. **Regenerate Tokens**: Refresh tokens every 24 hours when they expire
6. **Sequential Testing**: Use comma-separated tests for organized execution

## 🚀 Advanced Usage

### Direct k6 Execution

For granular control over individual test scripts:

```bash
# k6 Cloud execution (recommended for performance testing)
K6_ENVIRONMENT=DEV VUS=100 DURATION=5m \
k6 cloud run --env K6_ENVIRONMENT=DEV --env VUS=100 --env DURATION=5m core-api-load-test.js

# Local execution with cloud output (debugging)
K6_ENVIRONMENT=DEV VUS=10 DURATION=1m \
k6 run core-api-load-test.js --out cloud

# Local execution with JSON output (offline testing)
K6_ENVIRONMENT=DEV VUS=10 DURATION=1m \
k6 run core-api-load-test.js --out json=results.json
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