# AutoGPT Platform Load Testing

A comprehensive k6-based load testing suite for the AutoGPT Platform API with Grafana Cloud integration.

## 🎯 Overview

This testing suite provides:
- **API Load Testing**: Test REST API endpoints under various load conditions
- **WebSocket Testing**: Stress test real-time WebSocket connections
- **Stress Testing**: High-load scenarios to identify breaking points
- **Spike Testing**: Rapid load changes to test auto-scaling
- **Grafana Cloud Integration**: Advanced monitoring and alerting

## 📁 Project Structure

```
load-tests/
├── configs/
│   ├── environment.js                           # Environment configuration
│   └── grafana-cloud.js                         # Grafana Cloud setup
├── scenarios/
│   ├── comprehensive-platform-load-test.js      # Standard load testing
│   ├── high-concurrency-api-stress-test.js      # API stress testing
│   └── real-time-websocket-stress-test.js       # WebSocket testing
├── utils/
│   ├── auth.js                                  # Authentication utilities
│   └── test-data.js                             # Test data generators
├── data/
│   └── test-users.json                          # Test user configuration
├── results/                                     # Test results (local mode)
├── core-api-validation-test.js                  # Quick API validation (100% success)
├── comprehensive-error-diagnostic-test.js       # Detailed error analysis
├── core-api-success-validation-test.js          # Core API success verification
├── block-id-discovery-test.js                   # Block ID discovery utility
├── graph-creation-validation-test.js            # Graph creation validation
├── reduced-load-debugging-test.js               # Reduced load for debugging
├── graph-execution-load-test.js                # Dedicated graph execution testing
├── run-tests.sh                                 # Main test execution script
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

### Basic Usage

```bash
# Run standard load test
./run-tests.sh load

# Run stress test with custom parameters
./run-tests.sh stress -v 50 -d 5m

# Run all tests with Grafana Cloud integration
./run-tests.sh all --cloud

# Use environment variables for configuration
VUS=20 DURATION=5m ./run-tests.sh load

# Test specific functionality
k6 run graph-execution-load-test.js
k6 run core-api-validation-test.js
```

### ⚡ Environment Variable Configuration

All tests support easy configuration via environment variables:

```bash
# Basic load configuration
VUS=10                    # Number of virtual users
DURATION=2m               # Test duration
RAMP_UP=30s              # Ramp-up time
RAMP_DOWN=30s            # Ramp-down time

# Stress test configuration
STRESS_VUS=50            # Stress test VUs
STRESS_DURATION=5m       # Stress test duration

# Spike test configuration
SPIKE_VUS=100            # Spike test VUs
SPIKE_DURATION=30s       # Spike test duration

# Performance thresholds
THRESHOLD_P95=2000       # 95th percentile threshold (ms)
THRESHOLD_P99=5000       # 99th percentile threshold (ms)
THRESHOLD_ERROR_RATE=0.05 # Maximum error rate (5%)
THRESHOLD_RPS=10         # Minimum requests per second

# Environment targeting
K6_ENVIRONMENT=DEV       # DEV, STAGING, PROD
```

**Examples:**
```bash
# High-load stress test
STRESS_VUS=100 STRESS_DURATION=10m ./run-tests.sh stress

# Quick validation with custom thresholds
VUS=5 DURATION=30s THRESHOLD_P95=1000 k6 run core-api-validation-test.js

# Graph execution focused testing
VUS=3 DURATION=2m k6 run graph-execution-load-test.js
```

## 🔧 Configuration

### Environment Setup

Set your target environment:

```bash
# Test against dev environment (default)
export K6_ENVIRONMENT=DEV

# Test against staging
export K6_ENVIRONMENT=STAGING

# Test against production (be careful!)
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
   ./run-tests.sh load --cloud
   ```

## 📊 Test Types & Scenarios

### 🚀 Quick Validation Tests

#### Core API Validation (`core-api-validation-test.js`)
- **Purpose**: Fast validation that all APIs are working (recommended first test)
- **Duration**: ~10 seconds
- **Coverage**: Authentication, Profile, Credits, Graphs, Executions, Schedules, Onboarding
- **Expected Result**: 100% success rate

```bash
k6 run core-api-validation-test.js
```

#### Error Diagnostic (`comprehensive-error-diagnostic-test.js`)
- **Purpose**: Detailed analysis of any failing requests
- **Duration**: ~20 seconds  
- **Output**: Detailed error logs and response analysis
- **Use Case**: Debugging when tests fail

```bash
k6 run comprehensive-error-diagnostic-test.js
```

#### API Success Verification (`core-api-success-validation-test.js`)
- **Purpose**: Focused test to verify API success rates under moderate load
- **Duration**: 30 seconds with 5 VUs
- **Expected Result**: 100% success rate for core APIs

```bash
k6 run core-api-success-validation-test.js
```

### 🏋️ Load Testing Scenarios

#### 1. Comprehensive Platform Load Test (`comprehensive-platform-load-test.js`)
- **Purpose**: Full platform load testing with realistic user journeys
- **Default**: 10 VUs for 2 minutes
- **Tests**: Authentication, graph CRUD, block execution, system operations

```bash
k6 run scenarios/comprehensive-platform-load-test.js
# Or via script:
./run-tests.sh load -v 20 -d 5m
```

#### 2. High Concurrency API Stress Test (`high-concurrency-api-stress-test.js`)
- **Purpose**: Find system breaking points with high API load
- **Default**: 50 VUs for 5 minutes
- **Tests**: All API endpoints under maximum concurrent load

```bash
k6 run scenarios/high-concurrency-api-stress-test.js
# Or via script:
./run-tests.sh stress
```

#### 3. Real-time WebSocket Stress Test (`real-time-websocket-stress-test.js`)
- **Purpose**: Test WebSocket connections under load
- **Default**: 20 concurrent connections for 3 minutes
- **Tests**: WebSocket messaging, connection stability, real-time features

```bash
k6 run scenarios/real-time-websocket-stress-test.js
# Or via script:
./run-tests.sh websocket
```

### 🔧 Debugging & Utility Tests

#### Block ID Discovery (`block-id-discovery-test.js`)
- **Purpose**: Discover available blocks and their UUIDs
- **Use Case**: Finding correct block IDs for test data
- **Output**: Lists all available blocks with IDs and names

```bash
k6 run block-id-discovery-test.js
```

#### Graph Creation Validation (`graph-creation-validation-test.js`)
- **Purpose**: Test graph creation with correct block references
- **Use Case**: Validate graph creation logic
- **Tests**: Graph creation, validation, error handling

```bash
k6 run graph-creation-validation-test.js
```

#### Graph Execution Load Test (`graph-execution-load-test.js`) ⭐
- **Purpose**: Comprehensive graph creation and execution testing
- **Configuration**: Fully configurable via environment variables
- **Tests**: Graph creation, execution, monitoring, complex workflows
- **Features**: Simple and complex graph types, execution monitoring, error handling

```bash
# Basic graph execution testing
k6 run graph-execution-load-test.js

# High-load graph execution testing
VUS=10 DURATION=5m k6 run graph-execution-load-test.js

# Quick graph execution validation
VUS=2 DURATION=30s k6 run graph-execution-load-test.js
```

#### Reduced Load Debugging (`reduced-load-debugging-test.js`)
- **Purpose**: Run main load test with reduced parameters for debugging
- **Configuration**: 2 VUs for 15 seconds (vs normal 10 VUs for 2 minutes)
- **Use Case**: Debug load test issues without full load

```bash
k6 run reduced-load-debugging-test.js
```

### 📈 Advanced Testing Scenarios

#### 4. Spike Test (`spike`)
- **Purpose**: Test auto-scaling capabilities
- **Pattern**: Rapid ramp-up to 100 VUs, maintain, rapid ramp-down
- **Tests**: System responsiveness to traffic spikes

```bash
./run-tests.sh spike
```

#### 5. Complete Test Suite (`all`)
- **Purpose**: Comprehensive testing across all scenarios
- **Runs**: All test scenarios sequentially
- **Duration**: ~20 minutes total
- **Includes**: Load, stress, WebSocket, and spike tests

```bash
./run-tests.sh all --cloud
```

## 🔐 Test Data Setup

### 1. Create Test Users

Before running tests, create actual test accounts in your Supabase instance:

```bash
# Example: Create test users via Supabase CLI or dashboard
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

### Local Results

When running in local mode, results are saved to `results/` directory:

```bash
# View test summary
cat results/load_test_20231215_143022.json | jq '.metrics'

# Extract key metrics
jq '.metrics.http_req_duration.avg' results/load_test_*.json
```

### Grafana Cloud Dashboard

With cloud integration enabled, view results at:
- **Dashboard**: https://your-org.grafana.net/
- **Real-time monitoring**: Live test execution metrics
- **Alerts**: Automated notifications for threshold breaches

### Key Metrics to Monitor

1. **Performance**:
   - Response time (p95, p99)
   - Throughput (requests/second)
   - Error rate

2. **Business Logic**:
   - Authentication success rate
   - Graph creation/execution time
   - Block execution performance

3. **Infrastructure**:
   - CPU/Memory usage
   - Database performance
   - WebSocket connection stability

## 🚨 Alerting & Thresholds

Default SLA thresholds:
- **Response Time**: p95 < 2s, p99 < 5s
- **Error Rate**: < 5%
- **Throughput**: > 10 req/s
- **Authentication**: < 10% failure rate

Configure custom thresholds in `configs/grafana-cloud.js`.

## 🔍 Troubleshooting

### Common Issues

1. **Authentication Failures**:
   ```bash
   # Check test user credentials
   # Verify Supabase URL configuration
   # Ensure test users exist and are active
   ```

2. **Network Timeouts**:
   ```bash
   # Check connectivity to target environment
   # Verify DNS resolution
   # Test manual API calls with curl
   ```

3. **Insufficient Credits**:
   ```bash
   # Top up test user accounts
   # Check credit consumption rate
   # Consider using service accounts for testing
   ```

4. **Rate Limiting**:
   ```bash
   # Reduce VU count or increase ramp-up time
   # Check API rate limits
   # Use multiple test users
   ```

### Debug Mode

Run tests with increased verbosity:

```bash
# Enable debug logging
K6_LOG_LEVEL=debug ./run-tests.sh load

# Run single iteration for debugging
k6 run --vus 1 --iterations 1 scenarios/comprehensive-platform-load-test.js

# Quick API validation (recommended first test)
k6 run core-api-validation-test.js

# Detailed error diagnostics
k6 run comprehensive-error-diagnostic-test.js
```

## 🛡️ Security & Best Practices

### Security Guidelines

1. **Never use production credentials** for testing
2. **Use dedicated test environment** with isolated data
3. **Implement proper cleanup** procedures
4. **Monitor test costs** and credit consumption
5. **Rotate test credentials** regularly

### Performance Testing Best Practices

1. **Start small**: Begin with low VU counts
2. **Ramp gradually**: Use realistic ramp-up patterns
3. **Monitor resources**: Watch system metrics during tests
4. **Clean up data**: Remove test artifacts after testing
5. **Document baselines**: Track performance over time

## 📝 Contributing

### Adding New Test Scenarios

1. Create new test file in `scenarios/`
2. Follow existing patterns for authentication and metrics
3. Add configuration options to `run-tests.sh`
4. Update documentation

### Example Test Structure

```javascript
import { check } from 'k6';
import { getEnvironmentConfig } from '../configs/environment.js';
import { authenticateUser, getAuthHeaders } from '../utils/auth.js';

export const options = {
  stages: [
    { duration: '30s', target: 10 },
    { duration: '1m', target: 10 },
    { duration: '30s', target: 0 },
  ],
};

export default function() {
  // Your test logic here
}
```

## 📊 Example Commands

```bash
# Development testing
./run-tests.sh load -e DEV -v 10 -d 2m

# Staging stress test  
./run-tests.sh stress -e STAGING -v 50 -d 5m --cloud

# Production spike test (be careful!)
./run-tests.sh spike -e PROD --cloud

# WebSocket-specific testing
./run-tests.sh websocket -v 20 -d 3m

# Complete test suite with cloud monitoring
K6_ENVIRONMENT=DEV ./run-tests.sh all --cloud
```

## 🔗 Resources

- [k6 Documentation](https://k6.io/docs/)
- [Grafana Cloud k6](https://grafana.com/products/cloud/k6/)
- [AutoGPT Platform API Docs](https://dev-api.agpt.co/docs)
- [Performance Testing Best Practices](https://k6.io/docs/testing-guides/)

## 📞 Support

For issues with the load testing suite:
1. Check the troubleshooting section above
2. Review k6 and Grafana Cloud documentation
3. Open an issue in the project repository
4. Contact the platform team for environment-specific issues

---

**⚠️ Important**: Always coordinate load testing with the platform team, especially for staging and production environments. High-volume testing can impact other users and systems.