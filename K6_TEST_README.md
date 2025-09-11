# K6 Load Testing for AutoGPT Store API

This k6 script performs load testing on the AutoGPT Store API endpoints deployed on Vercel staging environment.

## Prerequisites

1. **Install k6**: 
   ```bash
   # macOS
   brew install k6
   
   # Windows
   choco install k6
   
   # Linux
   sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
   echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
   sudo apt-get update
   sudo apt-get install k6
   ```

2. **Obtain the Vercel Automation Bypass Secret**:
   - Go to your Vercel project settings
   - Navigate to "Security" → "Protection Bypass for Automation"
   - Copy the generated secret (or generate a new one if needed)
   - This secret allows bypassing Vercel Deployment Protection

## Running the Test

### Method 1: Using environment variable
```bash
export VERCEL_AUTOMATION_BYPASS_SECRET="your-secret-here"
k6 run backend-api-test-k6.js
```

### Method 2: Using command-line parameter
```bash
k6 run -e VERCEL_AUTOMATION_BYPASS_SECRET="your-secret-here" backend-api-test-k6.js
```

### Method 3: Using shorter alias
```bash
k6 run -e BYPASS_SECRET="your-secret-here" backend-api-test-k6.js
```

## Test Configuration

The test is configured with the following stages:
- Ramps up to 100 virtual users over 5 minutes
- Tests various Store API endpoints including:
  - `/agents` - List all agents with pagination
  - `/agents?featured=true` - Featured agents
  - `/agents?search_query=...` - Search functionality
  - `/agents/{username}/{agent_name}` - Individual agent details
  - `/creators` - List all creators
  - `/creator/{username}` - Creator profiles

## Performance Thresholds

The test will fail if:
- 90% of requests don't complete within 1 second
- 95% of requests don't complete within 2 seconds
- Error rate exceeds 5%

## Customizing the Test

### Shorter test run
```bash
k6 run -e VERCEL_AUTOMATION_BYPASS_SECRET="your-secret" --duration 1m --vus 10 backend-api-test-k6.js
```

### Different target URL
```bash
k6 run -e VERCEL_AUTOMATION_BYPASS_SECRET="your-secret" -e API_URL="https://your-deployment.vercel.app/api" backend-api-test-k6.js
```

## Viewing Results

After the test completes, k6 will display:
- Request rate (req/s)
- Request duration percentiles (p50, p90, p95)
- Error rate
- Data transferred
- Custom metrics for specific endpoints

## Troubleshooting

### 401 Unauthorized Error
- Verify the bypass secret is correct
- Check if the secret needs to be regenerated in Vercel settings
- Ensure you're using the correct environment (staging vs production)

### Connection Errors
- Verify the target URL is accessible
- Check if there are any network restrictions
- Ensure the Vercel deployment is active

## Important Notes

- The bypass secret is sensitive - never commit it to version control
- The secret needs to be updated if regenerated in Vercel settings
- For production testing, use appropriate rate limits to avoid overwhelming the service