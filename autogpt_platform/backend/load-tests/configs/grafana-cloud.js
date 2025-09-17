/**
 * Grafana Cloud K6 Configuration
 * 
 * This file contains configuration for integrating k6 tests with Grafana Cloud
 * for advanced monitoring, alerting, and dashboard visualization.
 */

export const GRAFANA_CLOUD_CONFIG = {
  // Grafana Cloud K6 Project Configuration
  PROJECT_ID: __ENV.K6_CLOUD_PROJECT_ID || '', // Set this in your environment
  TOKEN: __ENV.K6_CLOUD_TOKEN || '',           // Set this in your environment
  
  // Default test tags for organization
  DEFAULT_TAGS: {
    team: 'platform',
    service: 'autogpt-platform',
    environment: __ENV.K6_ENVIRONMENT || 'dev',
    version: __ENV.GIT_COMMIT || 'unknown',
    branch: __ENV.GIT_BRANCH || 'unknown',
  },
  
  // Test result retention (in days)
  RETENTION_DAYS: 30,
  
  // Alert thresholds for Grafana Cloud monitoring
  ALERT_THRESHOLDS: {
    // Performance thresholds
    response_time_p95: 2000,     // 95th percentile response time in ms
    response_time_p99: 5000,     // 99th percentile response time in ms
    error_rate: 0.05,            // Error rate threshold (5%)
    throughput_min: 10,          // Minimum requests per second
    
    // Resource thresholds
    cpu_usage_max: 80,           // Maximum CPU usage percentage
    memory_usage_max: 80,        // Maximum memory usage percentage
    disk_usage_max: 80,          // Maximum disk usage percentage
    
    // Business logic thresholds
    auth_error_rate: 0.1,        // Authentication error rate (10%)
    graph_creation_time: 5000,   // Maximum graph creation time in ms
    execution_time_max: 30000,   // Maximum execution time in ms (30s)
    
    // WebSocket specific thresholds
    ws_connection_error_rate: 0.1, // WebSocket connection error rate (10%)
    ws_message_latency: 2000,    // WebSocket message latency in ms
  },
  
  // Dashboard configuration
  DASHBOARD_CONFIG: {
    // Performance dashboard panels
    performance_panels: [
      'response_time_trends',
      'throughput_metrics',
      'error_rate_analysis',
      'resource_utilization'
    ],
    
    // Business metrics dashboard panels
    business_panels: [
      'user_authentication_metrics',
      'graph_operations_metrics', 
      'execution_success_rate',
      'api_endpoint_performance'
    ],
    
    // Infrastructure dashboard panels
    infrastructure_panels: [
      'kubernetes_metrics',
      'database_performance',
      'redis_metrics',
      'rabbitmq_metrics'
    ]
  },
  
  // Alert rules configuration
  ALERT_RULES: [
    {
      name: 'High Response Time',
      condition: 'avg(http_req_duration) > 2000',
      severity: 'warning',
      duration: '5m',
      message: 'API response times are above 2 seconds'
    },
    {
      name: 'High Error Rate', 
      condition: 'rate(http_req_failed) > 0.05',
      severity: 'critical',
      duration: '2m',
      message: 'Error rate is above 5%'
    },
    {
      name: 'Low Throughput',
      condition: 'rate(http_reqs) < 10',
      severity: 'warning', 
      duration: '5m',
      message: 'Request throughput is below 10 req/s'
    },
    {
      name: 'Authentication Failures',
      condition: 'rate(auth_errors) > 0.1',
      severity: 'critical',
      duration: '3m',
      message: 'Authentication error rate is above 10%'
    },
    {
      name: 'WebSocket Connection Issues',
      condition: 'rate(ws_connection_errors) > 0.1',
      severity: 'warning',
      duration: '5m', 
      message: 'WebSocket connection error rate is above 10%'
    }
  ],
  
  // Notification channels
  NOTIFICATION_CHANNELS: [
    {
      type: 'slack',
      url: __ENV.SLACK_WEBHOOK_URL || '',
      channel: '#platform-alerts',
      username: 'k6-monitoring'
    },
    {
      type: 'email',
      addresses: [
        __ENV.ALERT_EMAIL || 'platform-team@agpt.co'
      ]
    },
    {
      type: 'pagerduty',
      integration_key: __ENV.PAGERDUTY_INTEGRATION_KEY || ''
    }
  ]
};

/**
 * Generate cloud-specific test options
 */
export function getCloudTestOptions(testName, testType = 'load') {
  const baseOptions = {
    ext: {
      loadimpact: {
        projectID: GRAFANA_CLOUD_CONFIG.PROJECT_ID,
        name: testName,
        note: `Automated ${testType} test for AutoGPT Platform`,
      },
    },
    // Add cloud-specific tags
    tags: {
      ...GRAFANA_CLOUD_CONFIG.DEFAULT_TAGS,
      test_type: testType,
      test_name: testName,
    },
    // Cloud-specific thresholds with alerting
    thresholds: {
      http_req_duration: [
        `p(95)<${GRAFANA_CLOUD_CONFIG.ALERT_THRESHOLDS.response_time_p95}`,
        `p(99)<${GRAFANA_CLOUD_CONFIG.ALERT_THRESHOLDS.response_time_p99}`
      ],
      http_req_failed: [`rate<${GRAFANA_CLOUD_CONFIG.ALERT_THRESHOLDS.error_rate}`],
      http_reqs: [`rate>${GRAFANA_CLOUD_CONFIG.ALERT_THRESHOLDS.throughput_min}`],
      checks: ['rate>0.95'],
    }
  };
  
  return baseOptions;
}

/**
 * Setup cloud monitoring for test execution
 */
export function setupCloudMonitoring() {
  if (!GRAFANA_CLOUD_CONFIG.PROJECT_ID) {
    console.warn('⚠️  K6_CLOUD_PROJECT_ID not set - cloud features disabled');
    return false;
  }
  
  if (!GRAFANA_CLOUD_CONFIG.TOKEN) {
    console.warn('⚠️  K6_CLOUD_TOKEN not set - cloud features disabled');
    return false;
  }
  
  console.log('☁️  Grafana Cloud monitoring enabled');
  console.log(`📊 Project ID: ${GRAFANA_CLOUD_CONFIG.PROJECT_ID}`);
  console.log(`🏷️  Tags: ${JSON.stringify(GRAFANA_CLOUD_CONFIG.DEFAULT_TAGS)}`);
  
  return true;
}

/**
 * Send custom metrics to Grafana Cloud
 */
export function sendCustomMetric(metricName, value, tags = {}) {
  const metric = {
    name: metricName,
    value: value,
    timestamp: Date.now(),
    tags: {
      ...GRAFANA_CLOUD_CONFIG.DEFAULT_TAGS,
      ...tags
    }
  };
  
  // This would typically use the k6 cloud API or custom metric endpoint
  console.log(`📈 Custom metric: ${JSON.stringify(metric)}`);
}

/**
 * Generate test summary for cloud dashboard
 */
export function generateTestSummary(testResults) {
  return {
    test_name: testResults.testName || 'Unknown Test',
    start_time: testResults.startTime || Date.now(),
    end_time: testResults.endTime || Date.now(),
    duration: testResults.duration || 0,
    vus_max: testResults.vusMax || 0,
    iterations: testResults.iterations || 0,
    http_reqs: testResults.httpReqs || 0,
    http_req_duration_avg: testResults.httpReqDurationAvg || 0,
    http_req_duration_p95: testResults.httpReqDurationP95 || 0,
    http_req_failed_rate: testResults.httpReqFailedRate || 0,
    checks_passed_rate: testResults.checksPassedRate || 0,
    tags: GRAFANA_CLOUD_CONFIG.DEFAULT_TAGS
  };
}