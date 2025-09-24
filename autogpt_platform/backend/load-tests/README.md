# AutoGPT Platform Load Tests

Clean, comprehensive load testing infrastructure for the AutoGPT Platform using k6.

## 🚀 Quick Start

```bash
# 1. Set up Supabase service key (required for token generation)
export SUPABASE_SERVICE_KEY="your-supabase-service-key"

# 2. Generate pre-authenticated tokens (first time setup)  
node generate-tokens.js

# 3. Set up k6 cloud credentials (optional, for cloud testing)
cp configs/k6-credentials.env.example configs/k6-credentials.env
# Edit k6-credentials.env with your credentials from https://app.k6.io/

# 4. Verify all tests work locally
node run-tests.js verify

# 5. Run specific tests
node run-tests.js run core-api-test DEV local

# 6. Run cloud tests (requires k6 credentials)
node run-cloud-tests.js
```

## 📋 Available Tests

### 1. Core API Test (`core-api-load-test.js`)
- **Purpose**: Tests authenticated API endpoints that users interact with daily
- **Endpoints**: `/api/credits`, `/api/graphs`, `/api/blocks`, `/api/executions` 
- **Authentication**: Uses pre-authenticated tokens (no auth rate limiting)
- **Use Case**: Validates core platform API performance under load

### 2. Graph Execution Test (`graph-execution-load-test.js`) 
- **Purpose**: Tests the complete workflow creation and execution pipeline
- **Operations**: Graph creation, graph execution, result processing
- **Authentication**: Uses pre-authenticated tokens
- **Use Case**: Validates the platform's core automation capabilities under load

### 3. Marketplace Access Test (`marketplace-access-load-test.js`)
- **Purpose**: Tests public marketplace browsing without authentication
- **Endpoints**: `/api/store/agents`, `/api/store/featured`, public marketplace pages
- **Authentication**: None required (public endpoints)
- **Use Case**: Validates marketplace performance for anonymous visitors

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
- **File**: `configs/environment.js`
- **LOCAL**: `http://localhost:8006` (local development)
- **DEV**: `https://dev-api.agpt.co` (development environment)
- **PROD**: `https://api.agpt.co` (production environment)

## 📊 Test Modes

### Local Mode (Low Load)
- **Purpose**: Functional verification and basic performance testing
- **VUs**: 1-10 virtual users
- **Duration**: 30 seconds
- **Output**: JSON results saved to `results/` directory

### Cloud Mode (High Load) 
- **Purpose**: Comprehensive performance testing and scalability validation
- **VUs**: 20-100 virtual users
- **Duration**: 3 minutes
- **Output**: Results on k6 cloud dashboard + URLs saved to `k6-cloud-results.txt`

## ☸️ Kubernetes Pod Testing (Recommended for Stable Network)

For high-throughput testing with stable network conditions, run tests from within the Kubernetes cluster:

### 1. Deploy Load Test Pod

```bash
# Deploy the k6 load testing pod
kubectl apply -f k6-loadtesting-pod.yaml

# Verify pod is running
kubectl get pods -n dev-agpt | grep k6-loadtest
```

### 2. Copy Test Files to Pod

```bash
# Copy all load test files to pod
kubectl cp . dev-agpt/k6-loadtest-74d84cfc-fgv7l:/app

# Or copy specific files if needed
kubectl cp core-api-load-test.js dev-agpt/k6-loadtest-74d84cfc-fgv7l:/app/
kubectl cp configs/ dev-agpt/k6-loadtest-74d84cfc-fgv7l:/app/configs/
```

### 3. Generate Fresh Tokens in Pod

```bash
# Execute token generation from within the pod (uses service key from secrets)
kubectl exec -n dev-agpt k6-loadtest-74d84cfc-fgv7l -- node generate-tokens.js

# Verify tokens were created
kubectl exec -n dev-agpt k6-loadtest-74d84cfc-fgv7l -- ls -la configs/
```

### 4. Run Comprehensive Test Suite

```bash
# Run all 25 test scenarios from pod with stable network
kubectl exec -n dev-agpt k6-loadtest-74d84cfc-fgv7l -- node comprehensive-load-test-orchestrator.js

# Or run individual tests
kubectl exec -n dev-agpt k6-loadtest-74d84cfc-fgv7l -- \
  sh -c "K6_ENVIRONMENT=DEV VUS=100 DURATION=3m k6 run core-api-load-test.js"
```

### 5. Retrieve Results

```bash
# Copy results back to local machine
kubectl cp dev-agpt/k6-loadtest-74d84cfc-fgv7l:/app/comprehensive-results ./pod-results

# View CSV summary
kubectl exec -n dev-agpt k6-loadtest-74d84cfc-fgv7l -- \
  cat comprehensive-results/comprehensive_load_test_results.csv
```

### Pod Configuration Details

The `k6-loadtesting-pod.yaml` includes:

- **Node.js 18 Alpine**: Lightweight runtime environment
- **k6 v0.54.0**: Latest k6 version with modern JavaScript support
- **Resource Limits**: 2 CPU, 4Gi memory for high-throughput testing
- **Service Key Access**: Automatic access to Supabase service key from cluster secrets
- **k6 Cloud Integration**: Optional cloud testing with pre-configured credentials

### Advantages of Pod Testing

1. **Stable Network**: Eliminates local network variability and connection issues
2. **Higher Throughput**: Direct cluster networking provides optimal performance
3. **Resource Control**: Dedicated compute resources prevent local resource conflicts
4. **Security**: Service keys accessed securely from cluster secrets
5. **Consistency**: Reproducible test environment independent of local setup

## 🛠️ Usage

### List Available Tests
```bash
node run-tests.js list
```

### Run Individual Tests
```bash
# Local testing
node run-tests.js run core-api-test DEV local
node run-tests.js run graph-execution-test DEV local
node run-tests.js run marketplace-access-test DEV local

# Cloud testing  
node run-tests.js run core-api-test DEV cloud
node run-tests.js run graph-execution-test DEV cloud
node run-tests.js run marketplace-access-test DEV cloud
```

### Run All Tests
```bash
# Local testing (verification)
node run-tests.js run-all DEV local

# Cloud testing (full performance test)
node run-tests.js run-all DEV cloud
```

### Quick Verification
```bash
# Run all tests with minimal load to verify functionality
node run-tests.js verify
```

## 📈 Performance Expectations

Based on infrastructure analysis:

### Database Connection Limits
- **Current Capacity**: 9 pods × 20 connections = 180 total connections
- **Supabase Limit**: 200 connections (Pro plan)
- **Expected Ceiling**: ~102 RPS due to connection pool exhaustion
- **Per Pod**: ~51 RPS maximum throughput

### Authentication Performance  
- **Supabase Auth Rate Limit**: ~95 concurrent authentication requests
- **Solution**: Pre-authenticated tokens eliminate this bottleneck
- **Benefit**: Can now test actual API performance vs auth overhead

### Target Performance Metrics
- **P95 Latency**: < 10 seconds (authenticated endpoints)
- **P99 Latency**: < 20 seconds (authenticated endpoints)
- **Error Rate**: < 10% (allows for connection pool limits)
- **Success Rate**: > 85% (realistic under database constraints)

## 🔍 Troubleshooting

### Common Issues

**1. Authentication Failures**
```
❌ VU X has no valid pre-authenticated token
```
- **Solution**: Generate tokens using `node generate-tokens.js`

**2. Database Connection Errors**
```
Database connection timeout / pool exhausted
```
- **Expected**: At high load (100+ RPS) due to 20 connections per pod limit
- **Solution**: Test within expected 102 RPS ceiling

**3. k6 Cloud Failures**
```
Cloud test aborted / timeout
```
- **Solution**: Check `K6_CLOUD_TOKEN` and `K6_CLOUD_PROJECT_ID` environment variables

### Setup Requirements

**For local testing:**
```bash
# Set up Supabase service key (get from kubectl or environment)
export SUPABASE_SERVICE_KEY="your-supabase-service-key"

# Generate authentication tokens
node generate-tokens.js
```

**For k6 cloud testing:**
```bash
# Option 1: Use credentials file (recommended)
cp configs/k6-credentials.env.example configs/k6-credentials.env
# Edit with your credentials from https://app.k6.io/

# Option 2: Use environment variables
export K6_CLOUD_TOKEN="your-k6-cloud-token" 
export K6_CLOUD_PROJECT_ID="your-project-id"
```

## 📂 File Structure

```
load-tests/
├── README.md                              # This documentation
├── generate-tokens.js                     # Generate pre-auth tokens
├── run-tests.js                           # Local test runner  
├── run-cloud-tests.js                     # Cloud test runner
├── package.json                           # Node.js dependencies
├── .gitignore                             # Security exclusions
├── configs/
│   ├── environment.js                     # Environment URLs
│   ├── pre-authenticated-tokens.js        # 350+ tokens (gitignored)
│   ├── pre-authenticated-tokens.example.js # Token file template
│   ├── k6-credentials.env                 # k6 cloud creds (gitignored)  
│   └── k6-credentials.env.example         # Credentials template
├── core-api-load-test.js                  # Core API endpoints test
├── graph-execution-load-test.js           # Graph workflow test  
├── marketplace-access-load-test.js        # Public marketplace test
├── results/                               # Local test results (auto-created)
└── k6-cloud-results.txt                  # Cloud test URLs (auto-created)
```

## 🎯 Performance Testing Best Practices

1. **Start with Verification**: Always run `node run-tests.js verify` first
2. **Test Locally First**: Validate changes with local mode before cloud testing
3. **Respect Rate Limits**: Keep cloud tests within infrastructure capacity
4. **Monitor Results**: Check both application logs and k6 metrics
5. **Document Findings**: Save performance insights in test results

## 📚 Advanced Usage

### Custom Test Configurations

You can modify test parameters in `run-tests.js`:

```javascript
const TESTS = [
  {
    name: 'Core API Test',
    local_config: { VUS: 5, DURATION: '30s' },
    cloud_config: { VUS: 50, DURATION: '3m' }
  }
  // ... 
];
```

### Direct k6 Execution

For advanced users who want direct k6 control:

```bash
# Local test with custom parameters
K6_ENVIRONMENT=DEV VUS=10 DURATION=1m k6 run core-api-load-test.js

# Cloud test with custom parameters  
K6_ENVIRONMENT=DEV VUS=50 DURATION=5m \
K6_CLOUD_PROJECT_ID=your-id K6_CLOUD_TOKEN=your-token \
k6 run core-api-load-test.js --out cloud
```

---

## 🔗 Related Documentation

- [k6 Documentation](https://k6.io/docs/)
- [AutoGPT Platform API Documentation](https://docs.agpt.co/)
- [Supabase Connection Limits](https://supabase.com/docs/guides/database/connection-limits)

For questions or issues, please refer to the [AutoGPT Platform issues](https://github.com/Significant-Gravitas/AutoGPT/issues).