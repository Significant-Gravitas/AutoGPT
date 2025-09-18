# AutoGPT Platform Load Testing Infrastructure

Production-ready k6 load testing suite for the AutoGPT Platform API with Grafana Cloud integration.

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

### Basic Usage

```bash
# Run core API load test (recommended first test)
k6 run core-api-load-test.js

# Run graph execution load test
k6 run graph-execution-load-test.js

# Run comprehensive platform test
k6 run scenarios/comprehensive-platform-load-test.js

# Use environment variables for configuration
VUS=20 DURATION=5m k6 run core-api-load-test.js

# Run with Grafana Cloud monitoring
K6_CLOUD_TOKEN=your-token K6_CLOUD_PROJECT_ID=your-id k6 run core-api-load-test.js --out cloud
```

### ⚡ Environment Variable Configuration

All tests support easy configuration via environment variables:

```bash
# Basic load configuration
VUS=10                    # Number of virtual users
DURATION=2m               # Test duration
RAMP_UP=30s              # Ramp-up time
RAMP_DOWN=30s            # Ramp-down time

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
VUS=50 DURATION=10m k6 run comprehensive-platform-load-test.js

# Quick validation with custom thresholds
VUS=5 DURATION=30s THRESHOLD_P95=1000 k6 run core-api-load-test.js

# Graph execution focused testing
VUS=3 DURATION=2m k6 run graph-execution-load-test.js
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

### Validated Performance Metrics

Based on successful Grafana Cloud testing (Project ID: 4254406):

#### Core API Load Test ✅
- **Scale Tested**: 2-5 VUs, 30s-2m duration
- **Success Rate**: 100% for core API endpoints
- **Response Time**: p95 < 2s consistently
- **Recommended Production Scale**: 10-20 VUs for 5-10 minutes

#### Graph Execution Load Test ✅  
- **Scale Tested**: 2-5 VUs, 30s-3m duration
- **Success Rate**: 95%+ graph creation and execution
- **Complex Workflows**: Successfully tested multi-step graphs
- **Recommended Production Scale**: 5-10 VUs for sustained testing

#### Comprehensive Platform Test ✅
- **Scale Tested**: 2-10 VUs, 30s-2m duration  
- **Success Rate**: 100% check success, <5% HTTP failures
- **End-to-End Coverage**: Authentication through execution
- **Recommended Production Scale**: 10-15 VUs for realistic load

### Scale Recommendations for Production

**Development Testing:**
```bash
VUS=5 DURATION=2m k6 run [test-file] --out cloud
```

**Staging Validation:**
```bash
VUS=15 DURATION=5m k6 run [test-file] --out cloud
```

**Production Load Testing:**
```bash
VUS=25 DURATION=10m k6 run [test-file] --out cloud
```

**Stress Testing:**
```bash
VUS=50 DURATION=15m k6 run [test-file] --out cloud
```

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

1. **Performance**:
   - Response time (p95 < 2s, p99 < 5s)
   - Throughput (requests/second)
   - Error rate (< 5%)

2. **Business Logic**:
   - Authentication success rate (> 95%)
   - Graph creation/execution success rate (> 90%)
   - Block execution performance

3. **Infrastructure**:
   - CPU/Memory usage during tests
   - Database performance under load
   - API rate limiting behavior

## 🔍 Troubleshooting

### Common Issues

1. **Authentication Failures**:
   ```bash
   # Check test user credentials in data/test-users.json
   # Verify users exist in Supabase instance
   # Ensure SUPABASE_ANON_KEY is correct in configs/environment.js
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

## 📝 Example Commands

```bash
# Development testing
VUS=5 DURATION=2m k6 run core-api-load-test.js --out cloud

# Staging validation
VUS=15 DURATION=5m k6 run scenarios/comprehensive-platform-load-test.js --out cloud

# Graph-focused testing
VUS=8 DURATION=3m k6 run graph-execution-load-test.js --out cloud

# Quick API validation (recommended first test)
k6 run core-api-load-test.js

# Complete test suite
./run-tests.sh all --cloud
```

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