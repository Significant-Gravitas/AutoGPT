#!/usr/bin/env node

/**
 * Comprehensive Load Test Orchestrator
 * Runs all marketplace and library scenarios + core API tests sequentially
 * Provides detailed logging, error tracking, and CSV aggregation
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// All test scenarios as specified
const TEST_SCENARIOS = [
  // Marketplace Tests
  { name: "Viewing Marketplace (Logged Out)", type: "marketplace-access", day1_vus: 106, veryhigh_vus: 314, description: "Anonymous users browsing public marketplace" },
  { name: "Viewing Marketplace (Logged In)", type: "marketplace-library", day1_vus: 53, veryhigh_vus: 157, description: "Authenticated users browsing marketplace with library access" },
  
  // Library Management Tests
  { name: "Adding Agent to Library from Marketplace", type: "add-to-library", day1_vus: 32, veryhigh_vus: 95, description: "Users adding marketplace agents to personal library" },
  { name: "Viewing Library Home (0 Agents)", type: "library-empty", day1_vus: 53, veryhigh_vus: 157, description: "Users viewing empty personal library" },
  { name: "Viewing Library Home (1 Agent)", type: "library-single", day1_vus: 32, veryhigh_vus: 95, description: "Users viewing library with 1 agent" },
  { name: "Viewing Library Home (10 Agents)", type: "library-medium", day1_vus: 7, veryhigh_vus: 19, description: "Users viewing library with 10 agents" },
  { name: "Viewing Library Home (100 Agents)", type: "library-large", day1_vus: 2, veryhigh_vus: 4, description: "Users viewing library with 100 agents" },
  
  // Library Agent Viewing Tests  
  { name: "Viewing Library Agent (0 Runs)", type: "agent-no-runs", day1_vus: 53, veryhigh_vus: 157, description: "Users viewing agent with no execution history" },
  { name: "Viewing Library Agent (1 Run)", type: "agent-single-run", day1_vus: 32, veryhigh_vus: 95, description: "Users viewing agent with 1 execution" },
  { name: "Viewing Library Agent (10 Runs)", type: "agent-medium-runs", day1_vus: 20, veryhigh_vus: 57, description: "Users viewing agent with 10 executions" },
  { name: "Viewing Library Agent (100 Runs)", type: "agent-many-runs", day1_vus: 16, veryhigh_vus: 46, description: "Users viewing agent with 100 executions" }
];

// Additional core tests
const CORE_TESTS = [
  { name: "Core Graph Execution Test", type: "graph-execution", vus: 100, description: "Graph creation and execution pipeline testing" },
  { name: "Credits API Test", type: "credits-api", vus: 100, description: "User credits balance and spending API testing" },
  { name: "Marketplace API Test", type: "marketplace-api", vus: 100, description: "Marketplace search and browse API testing" }
];

class LoadTestOrchestrator {
  constructor() {
    this.results = [];
    this.currentTest = 0;
    this.totalTests = (TEST_SCENARIOS.length * 2) + CORE_TESTS.length; // Day1 + VeryHigh for scenarios + Core tests
    this.startTime = Date.now();
    
    // Create results directory
    this.resultsDir = path.join(process.cwd(), 'comprehensive-results');
    if (!fs.existsSync(this.resultsDir)) {
      fs.mkdirSync(this.resultsDir, { recursive: true });
    }
    
    console.log(`🚀 Starting Comprehensive Load Test Suite`);
    console.log(`📊 Total Tests: ${this.totalTests} tests`);
    console.log(`⏱️  Estimated Duration: ~2-3 hours`);
    console.log(`📂 Results Directory: ${this.resultsDir}`);
    console.log('=' .repeat(80));
  }

  async runTest(testConfig) {
    return new Promise((resolve) => {
      this.currentTest++;
      const progress = `[${this.currentTest}/${this.totalTests}]`;
      
      console.log(`\n${progress} 🔄 Starting: ${testConfig.name}`);
      console.log(`   📊 VUs: ${testConfig.vus}`);
      console.log(`   📝 Description: ${testConfig.description}`);
      console.log(`   ⏰ Started: ${new Date().toISOString()}`);
      
      const testStartTime = Date.now();
      const testEnv = {
        ...process.env,
        K6_ENVIRONMENT: 'DEV',
        VUS: testConfig.vus.toString(),
        DURATION: '3m',
        RAMP_UP: '30s', 
        RAMP_DOWN: '30s'
      };

      // Choose appropriate test file
      let testFile;
      if (testConfig.type.includes('marketplace')) {
        testFile = 'marketplace-access-load-test.js';
      } else if (testConfig.type === 'graph-execution') {
        testFile = 'graph-execution-load-test.js';
      } else if (testConfig.type.includes('api')) {
        testFile = 'core-api-load-test.js';
      } else {
        // For library tests, use core-api (we'll enhance it for library scenarios)
        testFile = 'core-api-load-test.js';
      }

      const outputFile = path.join(this.resultsDir, `${testConfig.name.replace(/[^a-zA-Z0-9]/g, '_')}_${testConfig.scenario}_${testConfig.vus}vus.json`);
      const summaryFile = path.join(this.resultsDir, `${testConfig.name.replace(/[^a-zA-Z0-9]/g, '_')}_${testConfig.scenario}_${testConfig.vus}vus_summary.json`);

      const k6Args = [
        'run', 
        testFile,
        '--out', `json=${outputFile}`,
        '--summary-export', summaryFile,
        '--quiet'
      ];

      const k6Process = spawn('k6', k6Args, {
        env: testEnv,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      let errorOutput = '';

      k6Process.stdout.on('data', (data) => {
        const text = data.toString();
        output += text;
        // Show key metrics during execution
        if (text.includes('http_reqs') || text.includes('http_req_failed') || text.includes('http_req_duration')) {
          process.stdout.write(`   📈 ${text.trim()}\n`);
        }
      });

      k6Process.stderr.on('data', (data) => {
        const text = data.toString();
        errorOutput += text;
        process.stderr.write(`   ❌ ${text}`);
      });

      k6Process.on('close', (code) => {
        const testDuration = Math.round((Date.now() - testStartTime) / 1000);
        
        // Parse results
        const result = this.parseTestResults(testConfig, outputFile, summaryFile, code, testDuration, errorOutput);
        this.results.push(result);
        
        const status = code === 0 ? '✅' : '❌';
        const totalDuration = Math.round((Date.now() - this.startTime) / 60000);
        
        console.log(`   ${status} Completed: ${testConfig.name}`);
        console.log(`   ⏱️  Duration: ${testDuration}s`);
        console.log(`   📊 RPS: ${result.rps}`);
        console.log(`   💥 Failure Rate: ${result.failure_rate}%`);
        console.log(`   🕐 P95 Latency: ${result.p95_latency}ms`);
        console.log(`   📈 Total Session Time: ${totalDuration}m`);
        
        resolve(result);
      });
    });
  }

  parseTestResults(testConfig, outputFile, summaryFile, exitCode, duration, errorOutput) {
    const result = {
      name: testConfig.name,
      scenario: testConfig.scenario,
      description: testConfig.description,
      vus: testConfig.vus,
      duration: duration,
      exit_code: exitCode,
      rps: 0,
      failure_rate: 0,
      total_requests: 0,
      failed_requests: 0,
      min_latency: 0,
      max_latency: 0,
      avg_latency: 0,
      p50_latency: 0,
      p90_latency: 0,
      p95_latency: 0,
      p99_latency: 0,
      error_details: '',
      path_failures: {}
    };

    try {
      // Parse summary file if exists
      if (fs.existsSync(summaryFile)) {
        const summaryData = JSON.parse(fs.readFileSync(summaryFile, 'utf8'));
        
        if (summaryData.metrics) {
          const metrics = summaryData.metrics;
          
          // HTTP requests
          if (metrics.http_reqs) {
            result.total_requests = metrics.http_reqs.count || 0;
            result.rps = Math.round(metrics.http_reqs.rate || 0);
          }
          
          // Failed requests  
          if (metrics.http_req_failed) {
            // http_req_failed.value is the failure rate (0-1), passes is successful requests  
            const failureRate = metrics.http_req_failed.value || 0;
            const successfulRequests = metrics.http_req_failed.passes || 0;
            
            result.failure_rate = Math.round(failureRate * 100);
            result.failed_requests = result.total_requests - successfulRequests;
          }
          
          // Latencies - k6 outputs in milliseconds directly
          if (metrics.http_req_duration) {
            const duration = metrics.http_req_duration;
            
            result.min_latency = Math.round(duration.min || 0);
            result.max_latency = Math.round(duration.max || 0);
            result.avg_latency = Math.round(duration.avg || 0);
            result.p50_latency = Math.round(duration.med || 0); // k6 uses 'med' for median/p50
            result.p90_latency = Math.round(duration['p(90)'] || 0);
            result.p95_latency = Math.round(duration['p(95)'] || 0);
            result.p99_latency = Math.round(duration['p(99)'] || 0);
          }
        }
      }

      // Parse detailed output file for path-specific failures
      if (fs.existsSync(outputFile)) {
        const outputData = fs.readFileSync(outputFile, 'utf8');
        const pathFailures = {};
        
        outputData.split('\n').forEach(line => {
          if (line.trim()) {
            try {
              const logEntry = JSON.parse(line);
              if (logEntry.type === 'Point' && logEntry.data && logEntry.data.tags) {
                const url = logEntry.data.tags.url;
                const status = logEntry.data.tags.status;
                const name = logEntry.data.tags.name;
                
                if (url && status && parseInt(status) >= 400) {
                  const key = `${name || url}`;
                  if (!pathFailures[key]) {
                    pathFailures[key] = { count: 0, statuses: {} };
                  }
                  pathFailures[key].count++;
                  pathFailures[key].statuses[status] = (pathFailures[key].statuses[status] || 0) + 1;
                }
              }
            } catch (e) {
              // Skip invalid JSON lines
            }
          }
        });
        
        result.path_failures = pathFailures;
      }

      // Add error output if any
      if (errorOutput) {
        result.error_details = errorOutput.substring(0, 500); // Limit error details
      }

    } catch (error) {
      result.error_details = `Failed to parse results: ${error.message}`;
    }

    return result;
  }

  async runAllTests() {
    console.log(`\n🎬 Starting Test Execution Sequence\n`);
    
    // Run all marketplace/library scenarios (Day 1 + Very High)
    for (const scenario of TEST_SCENARIOS) {
      // Day 1 scenario
      await this.runTest({
        ...scenario,
        vus: scenario.day1_vus,
        scenario: 'Day1'
      });
      
      // Very High scenario  
      await this.runTest({
        ...scenario,
        vus: scenario.veryhigh_vus,
        scenario: 'VeryHigh'
      });
    }

    // Run core tests
    for (const coreTest of CORE_TESTS) {
      await this.runTest({
        ...coreTest,
        scenario: 'Standard'
      });
    }

    console.log(`\n🎉 All tests completed!`);
    console.log(`📊 Generating comprehensive CSV report...`);
    
    this.generateCSVReport();
    this.printSummary();
  }

  generateCSVReport() {
    const csvHeaders = [
      'Test Name',
      'Scenario', 
      'Description',
      'VUs',
      'RPS',
      'Failure Rate (%)',
      'Total Requests',
      'Failed Requests',
      'Min Latency (ms)',
      'Max Latency (ms)', 
      'Avg Latency (ms)',
      'P50 Latency (ms)',
      'P90 Latency (ms)',
      'P95 Latency (ms)',
      'P99 Latency (ms)',
      'Duration (s)',
      'Path Failures',
      'Error Details'
    ];

    let csvContent = csvHeaders.join(',') + '\n';
    
    this.results.forEach(result => {
      const pathFailuresStr = Object.keys(result.path_failures).length > 0 
        ? JSON.stringify(result.path_failures).replace(/"/g, '""')
        : '';
      
      const row = [
        `"${result.name}"`,
        `"${result.scenario}"`, 
        `"${result.description}"`,
        result.vus,
        result.rps,
        result.failure_rate,
        result.total_requests,
        result.failed_requests,
        result.min_latency,
        result.max_latency,
        result.avg_latency,
        result.p50_latency,
        result.p90_latency,
        result.p95_latency,
        result.p99_latency,
        result.duration,
        `"${pathFailuresStr}"`,
        `"${result.error_details.replace(/"/g, '""')}"`
      ];
      
      csvContent += row.join(',') + '\n';
    });

    const csvFile = path.join(this.resultsDir, 'comprehensive_load_test_results.csv');
    fs.writeFileSync(csvFile, csvContent);
    
    console.log(`📄 CSV Report saved: ${csvFile}`);
  }

  printSummary() {
    const totalDuration = Math.round((Date.now() - this.startTime) / 60000);
    const successfulTests = this.results.filter(r => r.exit_code === 0).length;
    const failedTests = this.results.length - successfulTests;
    
    console.log(`\n📋 COMPREHENSIVE LOAD TEST SUMMARY`);
    console.log(`=====================================`);
    console.log(`⏱️  Total Duration: ${totalDuration} minutes`);
    console.log(`📊 Total Tests: ${this.results.length}`);
    console.log(`✅ Successful: ${successfulTests}`);
    console.log(`❌ Failed: ${failedTests}`);
    console.log(`📈 Average RPS: ${Math.round(this.results.reduce((sum, r) => sum + r.rps, 0) / this.results.length)}`);
    console.log(`💥 Average Failure Rate: ${Math.round(this.results.reduce((sum, r) => sum + r.failure_rate, 0) / this.results.length)}%`);
    console.log(`🕐 Average P95 Latency: ${Math.round(this.results.reduce((sum, r) => sum + r.p95_latency, 0) / this.results.length)}ms`);
    
    console.log(`\n🏆 TOP PERFORMERS:`);
    const topPerformers = this.results
      .filter(r => r.failure_rate < 5)
      .sort((a, b) => b.rps - a.rps)
      .slice(0, 3);
      
    topPerformers.forEach((result, i) => {
      console.log(`${i + 1}. ${result.name} (${result.scenario}): ${result.rps} RPS, ${result.failure_rate}% failures`);
    });

    console.log(`\n⚠️  AREAS OF CONCERN:`);
    const concerns = this.results
      .filter(r => r.failure_rate > 10 || r.p95_latency > 10000)
      .sort((a, b) => b.failure_rate - a.failure_rate)
      .slice(0, 5);
      
    concerns.forEach((result, i) => {
      console.log(`${i + 1}. ${result.name} (${result.scenario}): ${result.failure_rate}% failures, P95: ${result.p95_latency}ms`);
    });
    
    console.log(`\n📂 All detailed results saved in: ${this.resultsDir}`);
  }
}

// Start the orchestrator
if (require.main === module) {
  const orchestrator = new LoadTestOrchestrator();
  orchestrator.runAllTests().catch(console.error);
}

module.exports = LoadTestOrchestrator;