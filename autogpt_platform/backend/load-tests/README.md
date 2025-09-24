# AutoGPT Platform Load Tests

Clean, comprehensive load testing infrastructure for the AutoGPT Platform using k6.

## 🚀 Quick Start

```bash
# 1. Set up Supabase service key (required for token generation)
export SUPABASE_SERVICE_KEY="your-supabase-service-key"

# 2. Generate pre-authenticated tokens (first time setup - creates 350 tokens with 24-hour expiry)
node generate-tokens.js

# 3. Set up k6 cloud credentials (for cloud testing)
export K6_CLOUD_TOKEN="your-k6-cloud-token"
export K6_CLOUD_PROJECT_ID="4254406"

# 4. Run small validation tests (3 tests, ~5 minutes)
node load-test-orchestrator.js DEV cloud small

# 5. Run full comprehensive tests (25 tests, ~2 hours)  
node load-test-orchestrator.js DEV cloud full

# 6. Run tests locally (for debugging)
node load-test-orchestrator.js LOCAL local small
```

## 📋 Unified Load Test Orchestrator

The AutoGPT Platform now uses a unified orchestrator (`load-test-orchestrator.js`) that manages all load testing scenarios:

### Test Scales
- **small**: 3 validation tests (~5 minutes) - Quick functionality verification
- **full**: 25 comprehensive tests (~2 hours) - Complete performance analysis

### Test Categories (25 Total Tests)

#### 1. Marketplace Tests (8 tests)
- **Viewing_Marketplace_Logged_Out**: Public marketplace browsing (106-314 VUs)
- **Viewing_Marketplace_Logged_In**: Authenticated marketplace browsing (53-157 VUs)
- **Adding_Agent_to_Library**: Library management operations (32-95 VUs)
- **Viewing_Library_Home**: User library interface (53-157 VUs)

#### 2. Core API Tests (2 tests)
- **Core_API_Load_Test**: `/api/credits`, `/api/graphs`, `/api/blocks`, `/api/executions` (100 VUs)
- **Graph_Execution_Load_Test**: Complete workflow execution pipeline (100 VUs)

#### 3. Single Endpoint Tests (4 tests)
- Individual API endpoint stress testing: Credits, Graphs, Blocks, Executions (50 VUs each)

#### 4. Comprehensive Platform Tests (3 tests)
- **End-to-end user journeys**: Low (25), Medium (50), High (100 VUs)

#### 5. Stress Tests (2 tests)
- **Maximum load testing**: Marketplace (500 VUs), Core API (300 VUs)

#### 6. Extended Duration Tests (2 tests)
- **Long-duration sustainability**: 10-minute tests (100 VUs each)

#### 7. Authentication & Mixed Load Tests (4 tests)
- **User authentication workflows** and **mixed load patterns** (50-200 VUs)

## 🔧 Test Configuration

### Pre-Authenticated Tokens
- **Generation**: Run `node generate-tokens.js` to create tokens
- **File**: `configs/pre-authenticated-tokens.js` (gitignored for security)
- **Capacity**: 350+ tokens from 47 users
- **Expiry**: 24 hours (86400 seconds) - extended for long-duration testing
- **Benefit**: Eliminates Supabase auth rate limiting (95+ concurrent users)
- **Supports**: Up to 350 concurrent VUs without authentication bottlenecks
- **Security**: Tokens are gitignored and not committed to repository
- **Regeneration**: Run `node generate-tokens.js` when tokens expire after 24 hours

### Environment Configuration
- **LOCAL**: `http://localhost:8006` (local development)
- **DEV**: `https://dev-api.agpt.co` (development environment) 
- **PROD**: `https://api.agpt.co` (production environment - coordinate with team!)

## 📊 Test Modes

### Local Mode
- **Purpose**: Debugging and development testing
- **VUs**: 1-10 virtual users, short duration
- **Output**: JSON results saved to `results-*` directories

### Cloud Mode (Recommended)
- **Purpose**: Real performance testing and scalability validation
- **VUs**: 32-500 virtual users (varies by test)
- **Duration**: 3 minutes (10 minutes for extended tests)
- **Output**: Live k6 cloud dashboards + URLs captured automatically


## 🛠️ Usage

### Unified Load Test Orchestrator

The orchestrator supports different environments, execution modes, and test scales:

```bash
# Syntax: node load-test-orchestrator.js [ENVIRONMENT] [MODE] [SCALE]

# ENVIRONMENT: LOCAL | DEV | PROD
# MODE: local | cloud  
# SCALE: small | full
```

### Examples

```bash
# Quick validation (3 tests, local execution)
node load-test-orchestrator.js DEV local small

# Full test suite (25 tests, k6 cloud execution) - RECOMMENDED
node load-test-orchestrator.js DEV cloud full

# Local development testing
node load-test-orchestrator.js LOCAL local small

# Production testing (coordinate with team!)
node load-test-orchestrator.js PROD cloud small
```

### Help
```bash
# Show all available options and examples
node load-test-orchestrator.js --help
```

## 📈 Performance Expectations

### Current Validated Limits
- **Maximum Load Tested**: 314 VUs successfully handled ✅
- **Database Connection Pool**: ~102 RPS ceiling due to connection limits  
- **Authentication**: 350 concurrent users supported with pre-auth tokens
- **Library Workflows**: Fail around 157 VUs (marketplace-library tests)
- **Marketplace Browsing**: Scales well up to 314 VUs

### Target Performance Metrics  
- **P95 Latency**: < 60 seconds (current thresholds)
- **P99 Latency**: < 60 seconds (current thresholds)
- **Success Rate**: > 80% (allowing for infrastructure limits)
- **Execution Time**: ~5 minutes per test scenario

## 🔍 Troubleshooting

### Common Issues

**1. Authentication Failures**
```
❌ No valid authentication token available
```
- **Solution**: Run `node generate-tokens.js` to create fresh tokens (24-hour expiry)

**2. k6 Cloud Failures**  
```
Cloud test aborted or failed to start
```
- **Solution**: Verify `K6_CLOUD_TOKEN` and `K6_CLOUD_PROJECT_ID=4254406` are set correctly

**3. Test Failures (Exit code 99)**
```
❌ Test FAILED (exit code 99) 
```
- **Cause**: k6 threshold violations (P95/P99 response times exceeded)
- **Expected**: Some tests may fail under very high load (>150 VUs for library workflows)

### Required Setup

**1. Supabase Service Key (Required for all testing):**
```bash
# Get service key from Kubernetes secrets or environment
kubectl get secret dev-agpt-secrets -n dev-agpt -o jsonpath='{.data.SUPABASE_SERVICE_KEY}' | base64 -d

# Export for local use
export SUPABASE_SERVICE_KEY="your-supabase-service-key"
```

**2. Generate Pre-Authenticated Tokens (Required):**
```bash
# Creates 350 tokens with 24-hour expiry - prevents auth rate limiting  
node generate-tokens.js

# Regenerate when tokens expire (every 24 hours)
node generate-tokens.js
```

**3. k6 Cloud Credentials (Required for cloud testing):**
```bash
# Get credentials from https://app.k6.io/
export K6_CLOUD_TOKEN="your-k6-cloud-token" 
export K6_CLOUD_PROJECT_ID="4254406"  # AutoGPT Platform project ID
```

## 📂 File Structure

```
load-tests/
├── README.md                              # This documentation
├── load-test-orchestrator.js              # Unified test orchestrator (MAIN ENTRY POINT)
├── generate-tokens.js                     # Generate pre-auth tokens
├── package.json                           # Node.js dependencies
├── k6-loadtesting-pod.yaml               # Kubernetes pod for cluster testing
├── configs/
│   ├── environment.js                     # Environment URLs
│   ├── pre-authenticated-tokens.js        # 350+ tokens (gitignored)
│   ├── pre-authenticated-tokens.example.js # Token file template  
│   └── k6-credentials.env.example         # k6 cloud credentials template
├── data/
│   └── test-users.json                    # 47 test user accounts
├── utils/
│   ├── auth.js                            # Authentication utilities
│   └── test-data.js                       # Test data generation
├── Individual Test Scripts:
├── basic-connectivity-test.js             # Basic connectivity validation
├── core-api-load-test.js                  # Core API endpoints test
├── graph-execution-load-test.js           # Graph workflow execution test  
├── marketplace-access-load-test.js        # Public marketplace browsing
├── marketplace-library-load-test.js       # Authenticated marketplace/library
├── single-endpoint-test.js                # Individual API endpoint testing
├── scenarios/
│   └── comprehensive-platform-load-test.js # End-to-end user journeys
├── results-*/                             # Test results (auto-created)
└── *.txt                                  # Test URLs and logs (auto-created)
```

## 🎯 Performance Testing Best Practices

1. **Start with Small Scale**: Always run `node load-test-orchestrator.js DEV cloud small` first
2. **Test Locally for Development**: Use `local` mode for debugging and development
3. **Use k6 Cloud for Performance**: Use `cloud` mode for actual performance testing
4. **Monitor Real-Time**: Check k6 cloud dashboards during test execution
5. **Regenerate Tokens**: Run `node generate-tokens.js` every 24 hours when tokens expire

## 📚 Advanced Usage

### Custom Test Parameters

The orchestrator allows environment variables for fine-tuning:

```bash
# Custom VU and duration for specific tests
VUS=50 DURATION=5m RAMP_UP=1m RAMP_DOWN=1m \
node load-test-orchestrator.js DEV cloud small

# Single endpoint testing with high concurrency
ENDPOINT=blocks CONCURRENT_REQUESTS=100 VUS=10 DURATION=3m \
k6 run single-endpoint-test.js
```

### Direct k6 Execution (Advanced Users)

For granular control over individual test scripts:

```bash
# Run specific test with custom parameters
K6_ENVIRONMENT=DEV VUS=100 DURATION=3m \
K6_CLOUD_PROJECT_ID=4254406 K6_CLOUD_TOKEN=your-token \
k6 run core-api-load-test.js --out cloud

# Local execution with JSON output
K6_ENVIRONMENT=LOCAL VUS=10 DURATION=1m \
k6 run basic-connectivity-test.js --out json=results.json
```


---

## 🔗 Related Documentation

- [k6 Documentation](https://k6.io/docs/)
- [AutoGPT Platform API Documentation](https://docs.agpt.co/)
- [Supabase Connection Limits](https://supabase.com/docs/guides/database/connection-limits)

For questions or issues, please refer to the [AutoGPT Platform issues](https://github.com/Significant-Gravitas/AutoGPT/issues).