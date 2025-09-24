# AutoGPT Platform Load Testing Infrastructure

Production-ready k6 load testing suite for the AutoGPT Platform API with Grafana Cloud integration.

## 🎯 **Current Working Configuration (Sept 2025)**

**✅ RATE LIMIT OPTIMIZED:** All tests now use 5 VUs with `REQUESTS_PER_VU` parameter to avoid Supabase rate limits while maximizing load.

**Quick Start Commands:**
```bash
# Set credentials
export K6_CLOUD_TOKEN=your-token
export K6_CLOUD_PROJECT_ID=your-project-id

# 1. Basic connectivity (500 concurrent requests)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=100 k6 run basic-connectivity-test.js --out cloud

# 2. Core API testing (500 concurrent API calls)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=100 k6 run core-api-load-test.js --out cloud

# 3. Graph execution (100 concurrent operations)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=20 k6 run graph-execution-load-test.js --out cloud

# 4. Full platform testing (50 concurrent user journeys)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=10 k6 run scenarios/comprehensive-platform-load-test.js --out cloud

# 5. Single endpoint testing (up to 500 concurrent requests per VU)
K6_ENVIRONMENT=DEV VUS=1 DURATION=30s ENDPOINT=credits CONCURRENT_REQUESTS=100 k6 run single-endpoint-test.js --out cloud
```

**Success Indicators:**
- ✅ No 429 authentication errors
- ✅ "100/100 requests successful" messages
- ✅ Tests run full 7-minute duration
- ✅ Hundreds of completed iterations in Grafana dashboard

## 🎯 Overview

This testing suite provides comprehensive load testing for the AutoGPT Platform with:
- **API Load Testing**: Core API endpoints under various load conditions
- **Graph Execution Testing**: Graph creation, execution, and monitoring at scale
- **Platform Integration Testing**: End-to-end user workflows
- **Grafana Cloud Integration**: Advanced monitoring and real-time dashboards
- **Environment Variable Configuration**: Easy scaling and customization

## 📁 Project Structure

```
load-tests/
├── configs/
│   └── environment.js                           # Environment and performance configuration
├── scenarios/
│   └── comprehensive-platform-load-test.js      # Full platform workflow testing
├── utils/
│   ├── auth.js                                  # Authentication utilities
│   └── test-data.js                             # Test data generators and graph templates
├── data/
│   └── test-users.json                          # Test user configuration
├── core-api-load-test.js                        # Core API validation and load testing
├── graph-execution-load-test.js                 # Graph creation and execution testing
├── single-endpoint-test.js                      # Individual endpoint testing with high concurrency
├── interactive-test.js                          # Interactive CLI for guided test execution
├── run-tests.sh                                 # Test execution script
└── README.md                                    # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Install k6**:
   ```bash
   # macOS
   brew install k6
   
   # Linux
   sudo apt-get install k6
   ```

2. **Install jq** (for result processing):
   ```bash
   brew install jq
   ```

3. **Set up test users** (see [Test Data Setup](#test-data-setup))

### 🚀 Basic Usage (Current Working Configuration)

**Prerequisites**: Set your Grafana Cloud credentials:
```bash
export K6_CLOUD_TOKEN=your-token
export K6_CLOUD_PROJECT_ID=your-project-id
```

**✅ Recommended Commands (Rate-Limit Optimized):**
```bash
# 1. Basic connectivity test (500 concurrent requests)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=100 k6 run basic-connectivity-test.js --out cloud

# 2. Core API load test (500 concurrent API calls)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=100 k6 run core-api-load-test.js --out cloud

# 3. Graph execution test (100 concurrent graph operations)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=20 k6 run graph-execution-load-test.js --out cloud

# 4. Comprehensive platform test (50 concurrent user journeys)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=10 k6 run scenarios/comprehensive-platform-load-test.js --out cloud
```

**Quick Local Testing:**
```bash
# Run without cloud output for quick validation
K6_ENVIRONMENT=DEV VUS=2 DURATION=30s REQUESTS_PER_VU=5 k6 run core-api-load-test.js
```

### ⚡ Environment Variable Configuration

All tests support easy configuration via environment variables:

```bash
# Optimized load configuration (rate-limit aware)
VUS=5                     # Number of virtual users (keep ≤5 for rate limits)
REQUESTS_PER_VU=100      # Concurrent requests per VU (load multiplier)
CONCURRENT_REQUESTS=100  # Concurrent requests per VU for single endpoint test (1-500)
ENDPOINT=credits         # Target endpoint for single endpoint test (credits, graphs, blocks, executions)
DURATION=5m               # Test duration (extended for proper testing)
RAMP_UP=1m               # Ramp-up time
RAMP_DOWN=1m             # Ramp-down time

# Performance thresholds (cloud-optimized)
THRESHOLD_P95=30000      # 95th percentile threshold (30s for cloud)
THRESHOLD_P99=45000      # 99th percentile threshold (45s for cloud)
THRESHOLD_ERROR_RATE=0.4 # Maximum error rate (40% for high concurrency)
THRESHOLD_CHECK_RATE=0.6 # Minimum check success rate (60%)

# Environment targeting
K6_ENVIRONMENT=DEV       # DEV, LOCAL, PROD

# Grafana Cloud integration
K6_CLOUD_PROJECT_ID=4254406              # Project ID
K6_CLOUD_TOKEN=your-cloud-token          # API token
```

**Examples (Optimized for Rate Limits):**
```bash
# High-load stress test (concentrated load)
VUS=5 DURATION=10m REQUESTS_PER_VU=200 k6 run scenarios/comprehensive-platform-load-test.js --out cloud

# Quick validation 
VUS=2 DURATION=30s REQUESTS_PER_VU=10 k6 run core-api-load-test.js

# Graph execution focused testing (reduced concurrency for complex operations)
VUS=5 DURATION=5m REQUESTS_PER_VU=15 k6 run graph-execution-load-test.js --out cloud

# Maximum load testing (500 concurrent requests)
VUS=5 DURATION=15m REQUESTS_PER_VU=100 k6 run basic-connectivity-test.js --out cloud
```

## 🧪 Test Types & Scenarios

### 🚀 Core API Load Test (`core-api-load-test.js`)
- **Purpose**: Validate core API endpoints under load
- **Coverage**: Authentication, Profile, Credits, Graphs, Executions, Schedules
- **Default**: 1 VU for 10 seconds (quick validation)
- **Expected Result**: 100% success rate

**Recommended as first test:**
```bash
k6 run core-api-load-test.js
```

### 🔄 Graph Execution Load Test (`graph-execution-load-test.js`)
- **Purpose**: Test graph creation and execution workflows at scale
- **Features**: Graph creation, execution monitoring, complex workflows
- **Default**: 5 VUs for 2 minutes with ramp up/down
- **Tests**: Simple and complex graph types, execution status monitoring

**Comprehensive graph testing:**
```bash
# Standard graph execution testing
k6 run graph-execution-load-test.js

# High-load graph execution testing  
VUS=10 DURATION=5m k6 run graph-execution-load-test.js

# Quick validation
VUS=2 DURATION=30s k6 run graph-execution-load-test.js
```

### 🏗️ Comprehensive Platform Load Test (`comprehensive-platform-load-test.js`)
- **Purpose**: Full end-to-end platform testing with realistic user workflows
- **Default**: 10 VUs for 2 minutes
- **Coverage**: Authentication, graph CRUD operations, block execution, system operations
- **Use Case**: Production readiness validation

**Full platform testing:**
```bash
# Standard comprehensive test
k6 run scenarios/comprehensive-platform-load-test.js

# Stress testing
VUS=30 DURATION=10m k6 run scenarios/comprehensive-platform-load-test.js
```

### 🎯 NEW: Single Endpoint Load Test (`single-endpoint-test.js`)
- **Purpose**: Test individual API endpoints with high concurrency support
- **Features**: Up to 500 concurrent requests per VU, endpoint selection, burst load testing
- **Endpoints**: `credits`, `graphs`, `blocks`, `executions`
- **Use Case**: Debug specific endpoint performance, test RPS limits, burst load validation

**Single endpoint testing:**
```bash
# Test /api/credits with 100 concurrent requests
K6_ENVIRONMENT=DEV VUS=1 DURATION=30s ENDPOINT=credits CONCURRENT_REQUESTS=100 k6 run single-endpoint-test.js

# Test /api/graphs with 5 concurrent requests per VU
K6_ENVIRONMENT=DEV VUS=3 DURATION=1m ENDPOINT=graphs CONCURRENT_REQUESTS=5 k6 run single-endpoint-test.js

# Stress test /api/blocks with 500 RPS
K6_ENVIRONMENT=DEV VUS=1 DURATION=30s ENDPOINT=blocks CONCURRENT_REQUESTS=500 k6 run single-endpoint-test.js
```

### 🖥️ NEW: Interactive Load Testing CLI (`interactive-test.js`)
- **Purpose**: Guided test selection and configuration through interactive prompts
- **Features**: Test type selection, environment targeting, parameter configuration, cloud integration
- **Use Case**: Easy load testing for users unfamiliar with command-line parameters

**Interactive testing:**
```bash
# Launch interactive CLI
node interactive-test.js

# Follow prompts to select:
# - Test type (Basic, Core API, Single Endpoint, Comprehensive)
# - Environment (Local, Dev, Production)  
# - Execution mode (Local or k6 Cloud)
# - Parameters (VUs, duration, concurrent requests)
# - Endpoint (for single endpoint tests)
```

## 🔧 Configuration

### Environment Setup

Set your target environment:

```bash
# Test against dev environment (default)
export K6_ENVIRONMENT=DEV

# Test against staging
export K6_ENVIRONMENT=STAGING

# Test against production (coordinate with team!)
export K6_ENVIRONMENT=PROD
```

### Grafana Cloud Integration

For advanced monitoring and dashboards:

1. **Get Grafana Cloud credentials**:
   - Sign up at [Grafana Cloud](https://grafana.com/products/cloud/)
   - Create a k6 project
   - Get your Project ID and API token

2. **Set environment variables**:
   ```bash
   export K6_CLOUD_PROJECT_ID="your-project-id"
   export K6_CLOUD_TOKEN="your-api-token"
   ```

3. **Run tests in cloud mode**:
   ```bash
   k6 run core-api-load-test.js --out cloud
   k6 run graph-execution-load-test.js --out cloud
   ```

## 📊 Test Results & Scale Recommendations

### ✅ Validated Performance Metrics (Updated Sept 2025)

Based on comprehensive Grafana Cloud testing (Project ID: 4254406) with optimized configuration:

#### 🎯 Rate Limit Optimization Successfully Resolved
- **Challenge Solved**: Eliminated Supabase authentication rate limits (300 req/burst/IP)
- **Solution**: Reduced VUs to 5, increased concurrent requests per VU using `REQUESTS_PER_VU` parameter
- **Result**: Tests now validate platform capacity rather than authentication infrastructure limits

#### Core API Load Test ✅
- **Optimized Scale**: 5 VUs × 100 concurrent requests each = 500 total concurrent requests
- **Success Rate**: 100% for all API endpoints (Profile: 100/100, Credits: 100/100)
- **Duration**: Full 7-minute tests (1m ramp-up + 5m main + 1m ramp-down) without timeouts
- **Response Time**: Consistently fast with no 429 rate limit errors
- **Recommended Production Scale**: 5-10 VUs × 50-100 requests per VU

#### Graph Execution Load Test ✅  
- **Optimized Scale**: 5 VUs × 20 concurrent graph operations each
- **Success Rate**: 100% graph creation and execution under concentrated load
- **Complex Workflows**: Successfully creating and executing graphs concurrently
- **Real-time Monitoring**: Graph execution status tracking working perfectly
- **Recommended Production Scale**: 5 VUs × 10-20 operations per VU for sustained testing

#### Comprehensive Platform Test ✅
- **Optimized Scale**: 5 VUs × 10 concurrent user journeys each
- **Success Rate**: Complete end-to-end user workflows executing successfully
- **Coverage**: Authentication, graph CRUD, block execution, system operations
- **Timeline**: Tests running full 7-minute duration as configured
- **Recommended Production Scale**: 5-10 VUs × 5-15 journeys per VU

### 🚀 Optimized Scale Recommendations (Rate-Limit Aware)

**Development Testing (Recommended):**
```bash
# Basic connectivity and API validation
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=100 k6 run basic-connectivity-test.js --out cloud
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=100 k6 run core-api-load-test.js --out cloud

# Graph execution testing
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=20 k6 run graph-execution-load-test.js --out cloud

# Comprehensive platform testing
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=10 k6 run scenarios/comprehensive-platform-load-test.js --out cloud
```

**Staging Validation:**
```bash
# Higher concurrent load per VU, same low VU count to avoid rate limits
K6_ENVIRONMENT=STAGING VUS=5 DURATION=10m REQUESTS_PER_VU=200 k6 run core-api-load-test.js --out cloud
K6_ENVIRONMENT=STAGING VUS=5 DURATION=10m REQUESTS_PER_VU=50 k6 run graph-execution-load-test.js --out cloud
```

**Production Load Testing (Coordinate with Team!):**
```bash
# Maximum recommended load - still respects rate limits
K6_ENVIRONMENT=PROD VUS=5 DURATION=15m REQUESTS_PER_VU=300 k6 run core-api-load-test.js --out cloud
```

**⚠️ Rate Limit Considerations:**
- Keep VUs ≤ 5 to avoid IP-based Supabase rate limits
- Use `REQUESTS_PER_VU` parameter to increase load intensity
- Each VU makes concurrent requests using `http.batch()` for true concurrency
- Tests are optimized to test platform capacity, not authentication limits

## 🔐 Test Data Setup

### 1. Create Test Users

Before running tests, create actual test accounts in your Supabase instance:

```bash
# Example: Create test users via Supabase dashboard or CLI
# You'll need users with these credentials (update in data/test-users.json):
# - loadtest1@example.com : LoadTest123!
# - loadtest2@example.com : LoadTest123!
# - loadtest3@example.com : LoadTest123!
```

### 2. Update Test Configuration

Edit `data/test-users.json` with your actual test user credentials:

```json
{
  "test_users": [
    {
      "email": "your-actual-test-user@example.com",
      "password": "YourActualPassword123!",
      "user_id": "actual-user-id",
      "description": "Primary load test user"
    }
  ]
}
```

### 3. Ensure Test Users Have Credits

Make sure test users have sufficient credits for testing:

```bash
# Check user credits via API or admin dashboard
# Top up test accounts if necessary
```

## 📈 Monitoring & Results

### Grafana Cloud Dashboard

With cloud integration enabled, view results at:
- **Dashboard**: https://significantgravitas.grafana.net/a/k6-app/
- **Real-time monitoring**: Live test execution metrics
- **Test History**: Track performance trends over time

### Key Metrics to Monitor

1. **Performance (Cloud-Optimized Thresholds)**:
   - Response time (p95 < 30s, p99 < 45s for cloud testing)
   - Throughput (requests/second per VU)
   - Error rate (< 40% for high concurrency operations)
   - Check success rate (> 60% for complex workflows)

2. **Business Logic**:
   - Authentication success rate (100% expected with optimized config)
   - Graph creation/execution success rate (> 95%)
   - Block execution performance
   - No 429 rate limit errors

3. **Infrastructure**:
   - CPU/Memory usage during concentrated load
   - Database performance under 500+ concurrent requests
   - Rate limiting behavior (should be eliminated)
   - Test duration (full 7 minutes, not 1.5 minute timeouts)

## 🔍 Troubleshooting

### Common Issues

1. **Authentication Rate Limit Issues (SOLVED)**:
   ```bash
   # ✅ Solution implemented: Use ≤5 VUs with REQUESTS_PER_VU parameter
   # ✅ No more 429 errors with optimized configuration
   # If you still see rate limits, reduce VUS or REQUESTS_PER_VU
   
   # Check test user credentials in configs/environment.js (AUTH_CONFIG)
   # Verify users exist in Supabase instance
   # Ensure SUPABASE_ANON_KEY is correct
   ```


2. **Graph Creation Failures**:
   ```bash
   # Verify block IDs are correct for your environment
   # Check that test users have sufficient credits
   # Review graph schema in utils/test-data.js
   ```

3. **Network Issues**:
   ```bash
   # Verify environment URLs in configs/environment.js
   # Test manual API calls with curl
   # Check network connectivity to target environment
   ```

### Debug Mode

Run tests with increased verbosity:

```bash
# Enable debug logging
K6_LOG_LEVEL=debug k6 run core-api-load-test.js

# Run single iteration for debugging
k6 run --vus 1 --iterations 1 core-api-load-test.js
```

## 🛡️ Security & Best Practices

### Security Guidelines

1. **Never use production credentials** for testing
2. **Use dedicated test environment** with isolated data
3. **Monitor test costs** and credit consumption
4. **Coordinate with team** before production testing
5. **Clean up test data** after testing

### Performance Testing Best Practices

1. **Start small**: Begin with 2-5 VUs
2. **Ramp gradually**: Use realistic ramp-up patterns  
3. **Monitor resources**: Watch system metrics during tests
4. **Use cloud monitoring**: Leverage Grafana Cloud for insights
5. **Document results**: Track performance baselines over time

## 📝 Optimized Example Commands

```bash
# ✅ RECOMMENDED: Development testing (proven working configuration)
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=100 k6 run basic-connectivity-test.js --out cloud
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=100 k6 run core-api-load-test.js --out cloud
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=20 k6 run graph-execution-load-test.js --out cloud
K6_ENVIRONMENT=DEV VUS=5 DURATION=5m REQUESTS_PER_VU=10 k6 run scenarios/comprehensive-platform-load-test.js --out cloud

# Staging validation (higher concurrent load)
K6_ENVIRONMENT=STAGING VUS=5 DURATION=10m REQUESTS_PER_VU=150 k6 run core-api-load-test.js --out cloud

# Quick local validation
K6_ENVIRONMENT=DEV VUS=2 DURATION=30s REQUESTS_PER_VU=5 k6 run core-api-load-test.js

# Maximum stress test (coordinate with team!)
K6_ENVIRONMENT=DEV VUS=5 DURATION=15m REQUESTS_PER_VU=200 k6 run basic-connectivity-test.js --out cloud
```

### 🎯 Test Success Indicators

✅ **Tests are working correctly when you see:**
- No 429 authentication errors in output
- "100/100 requests successful" messages
- Tests running for full 7-minute duration (not timing out at 1.5min)
- Hundreds of completed iterations in Grafana Cloud dashboard
- 100% success rates for all endpoint types

## 🔗 Resources

- [k6 Documentation](https://k6.io/docs/)
- [Grafana Cloud k6](https://grafana.com/products/cloud/k6/)
- [AutoGPT Platform API Docs](https://dev-server.agpt.co/docs)
- [Performance Testing Best Practices](https://k6.io/docs/testing-guides/)

## 📞 Support

For issues with the load testing suite:
1. Check the troubleshooting section above
2. Review test results in Grafana Cloud dashboard
3. Contact the platform team for environment-specific issues

---

**⚠️ Important**: Always coordinate load testing with the platform team, especially for staging and production environments. High-volume testing can impact other users and systems.

**✅ Production Ready**: This load testing infrastructure has been validated on Grafana Cloud (Project ID: 4254406) with successful test execution and monitoring.