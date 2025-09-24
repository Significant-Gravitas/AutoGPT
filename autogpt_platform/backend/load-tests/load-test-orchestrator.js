#!/usr/bin/env node

// AutoGPT Platform Load Test Orchestrator
// Runs comprehensive test suite locally or in k6 cloud
// Collects URLs, statistics, and generates reports

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🎯 AUTOGPT PLATFORM LOAD TEST ORCHESTRATOR\n');
console.log('===========================================\n');

// Parse command line arguments
const args = process.argv.slice(2);
const environment = args[0] || 'DEV';       // LOCAL, DEV, PROD
const executionMode = args[1] || 'cloud';   // local, cloud  
const testScale = args[2] || 'full';        // small, full

console.log(`🌍 Target Environment: ${environment}`);
console.log(`🚀 Execution Mode: ${executionMode}`);
console.log(`📏 Test Scale: ${testScale}`);

// Test scenario definitions
const testScenarios = {
  // Small scale for validation (3 tests, ~5 minutes)
  small: [
    { name: "Basic_Connectivity_Test", file: "basic-connectivity-test.js", vus: 5, duration: "30s" },
    { name: "Core_API_Quick_Test", file: "core-api-load-test.js", vus: 10, duration: "1m" },
    { name: "Marketplace_Quick_Test", file: "marketplace-access-load-test.js", vus: 15, duration: "1m" }
  ],
  
  // Full comprehensive test suite (25 tests, ~2 hours)
  full: [
    // Marketplace Viewing Tests
    { name: "Viewing_Marketplace_Logged_Out_Day1", file: "marketplace-access-load-test.js", vus: 106, duration: "3m" },
    { name: "Viewing_Marketplace_Logged_Out_VeryHigh", file: "marketplace-access-load-test.js", vus: 314, duration: "3m" },
    { name: "Viewing_Marketplace_Logged_In_Day1", file: "marketplace-library-load-test.js", vus: 53, duration: "3m" },
    { name: "Viewing_Marketplace_Logged_In_VeryHigh", file: "marketplace-library-load-test.js", vus: 157, duration: "3m" },
    
    // Library Management Tests  
    { name: "Adding_Agent_to_Library_Day1", file: "marketplace-library-load-test.js", vus: 32, duration: "3m" },
    { name: "Adding_Agent_to_Library_VeryHigh", file: "marketplace-library-load-test.js", vus: 95, duration: "3m" },
    { name: "Viewing_Library_Home_0_Agents_Day1", file: "marketplace-library-load-test.js", vus: 53, duration: "3m" },
    { name: "Viewing_Library_Home_0_Agents_VeryHigh", file: "marketplace-library-load-test.js", vus: 157, duration: "3m" },

    // Core API Tests
    { name: "Core_API_Load_Test", file: "core-api-load-test.js", vus: 100, duration: "3m" },
    { name: "Graph_Execution_Load_Test", file: "graph-execution-load-test.js", vus: 100, duration: "3m" },
    
    // Single API Endpoint Tests  
    { name: "Credits_API_Single_Endpoint", file: "single-endpoint-test.js", vus: 50, duration: "3m", 
      env: { ENDPOINT: "credits", CONCURRENT_REQUESTS: 10 } },
    { name: "Graphs_API_Single_Endpoint", file: "single-endpoint-test.js", vus: 50, duration: "3m", 
      env: { ENDPOINT: "graphs", CONCURRENT_REQUESTS: 10 } },
    { name: "Blocks_API_Single_Endpoint", file: "single-endpoint-test.js", vus: 50, duration: "3m", 
      env: { ENDPOINT: "blocks", CONCURRENT_REQUESTS: 10 } },
    { name: "Executions_API_Single_Endpoint", file: "single-endpoint-test.js", vus: 50, duration: "3m", 
      env: { ENDPOINT: "executions", CONCURRENT_REQUESTS: 10 } },
    
    // Comprehensive Platform Tests
    { name: "Comprehensive_Platform_Low", file: "scenarios/comprehensive-platform-load-test.js", vus: 25, duration: "3m" },
    { name: "Comprehensive_Platform_Medium", file: "scenarios/comprehensive-platform-load-test.js", vus: 50, duration: "3m" },
    { name: "Comprehensive_Platform_High", file: "scenarios/comprehensive-platform-load-test.js", vus: 100, duration: "3m" },
    
    // User Authentication Workflows
    { name: "User_Auth_Workflows_Day1", file: "basic-connectivity-test.js", vus: 50, duration: "3m" },
    { name: "User_Auth_Workflows_VeryHigh", file: "basic-connectivity-test.js", vus: 100, duration: "3m" },

    // Mixed Load Tests
    { name: "Mixed_Load_Light", file: "core-api-load-test.js", vus: 75, duration: "5m" },
    { name: "Mixed_Load_Heavy", file: "marketplace-access-load-test.js", vus: 200, duration: "5m" },
    
    // Stress Tests
    { name: "Marketplace_Stress_Test", file: "marketplace-access-load-test.js", vus: 500, duration: "3m" },
    { name: "Core_API_Stress_Test", file: "core-api-load-test.js", vus: 300, duration: "3m" },
    
    // Extended Duration Tests
    { name: "Long_Duration_Marketplace", file: "marketplace-library-load-test.js", vus: 100, duration: "10m" },
    { name: "Long_Duration_Core_API", file: "core-api-load-test.js", vus: 100, duration: "10m" }
  ]
};

const scenarios = testScenarios[testScale];
console.log(`📊 Running ${scenarios.length} test scenarios`);

// Results collection
const results = [];
const cloudUrls = [];
const detailedMetrics = [];

// Create results directory  
const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 16);
const resultsDir = `results-${environment.toLowerCase()}-${executionMode}-${testScale}-${timestamp}`;
if (!fs.existsSync(resultsDir)) {
  fs.mkdirSync(resultsDir);
}

// Function to run a single test
function runTest(scenario, testIndex) {
  return new Promise((resolve, reject) => {
    console.log(`\n🚀 Test ${testIndex}/${scenarios.length}: ${scenario.name}`);
    console.log(`📊 Config: ${scenario.vus} VUs × ${scenario.duration} (${executionMode} mode)`);
    console.log(`📁 Script: ${scenario.file}`);
    
    // Build k6 command
    let k6Command, k6Args;
    
    // Determine k6 binary location
    const isInPod = fs.existsSync('/app/k6-v0.54.0-linux-amd64/k6');
    const k6Binary = isInPod ? '/app/k6-v0.54.0-linux-amd64/k6' : 'k6';
    
    // Build environment variables
    const envVars = [
      `K6_ENVIRONMENT=${environment}`,
      `VUS=${scenario.vus}`,
      `DURATION=${scenario.duration}`,
      `RAMP_UP=30s`,
      `RAMP_DOWN=30s`,
      `THRESHOLD_P95=60000`,
      `THRESHOLD_P99=60000`
    ];
    
    // Add scenario-specific environment variables
    if (scenario.env) {
      Object.keys(scenario.env).forEach(key => {
        envVars.push(`${key}=${scenario.env[key]}`);
      });
    }
    
    // Configure command based on execution mode
    if (executionMode === 'cloud') {
      k6Command = k6Binary;
      k6Args = ['cloud', 'run', scenario.file];
      // Add environment variables as --env flags
      envVars.forEach(env => {
        k6Args.push('--env', env);
      });
    } else {
      k6Command = k6Binary;  
      k6Args = ['run', scenario.file];
      
      // Add local output files
      const outputFile = path.join(resultsDir, `${scenario.name}.json`);
      const summaryFile = path.join(resultsDir, `${scenario.name}_summary.json`);
      k6Args.push('--out', `json=${outputFile}`);
      k6Args.push('--summary-export', summaryFile);
    }
    
    const startTime = Date.now();
    let testUrl = '';
    let stdout = '';
    let stderr = '';
    
    console.log(`⏱️ Test started: ${new Date().toISOString()}`);
    
    // Set environment variables for spawned process
    const processEnv = { ...process.env };
    envVars.forEach(env => {
      const [key, value] = env.split('=');
      processEnv[key] = value;
    });
    
    const childProcess = spawn(k6Command, k6Args, {
      env: processEnv,
      stdio: ['ignore', 'pipe', 'pipe']
    });
    
    // Handle stdout
    childProcess.stdout.on('data', (data) => {
      const output = data.toString();
      stdout += output;
      
      // Extract k6 cloud URL
      if (executionMode === 'cloud') {
        const urlMatch = output.match(/output:\s*(https:\/\/[^\s]+)/);
        if (urlMatch) {
          testUrl = urlMatch[1];
          console.log(`🔗 Test URL: ${testUrl}`);
        }
      }
      
      // Show progress indicators
      if (output.includes('Run    [')) {
        const progressMatch = output.match(/Run\s+\[\s*(\d+)%\s*\]/);
        if (progressMatch) {
          process.stdout.write(`\r⏳ Progress: ${progressMatch[1]}%`);
        }
      }
    });
    
    // Handle stderr
    childProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    // Handle process completion
    childProcess.on('close', (code) => {
      const endTime = Date.now();
      const duration = Math.round((endTime - startTime) / 1000);
      
      console.log(`\n⏱️ Completed in ${duration}s`);
      
      if (code === 0) {
        console.log(`✅ ${scenario.name} SUCCESS`);
        
        const result = {
          test: scenario.name,
          status: 'SUCCESS',
          duration: `${duration}s`,
          vus: scenario.vus,
          target_duration: scenario.duration,
          url: testUrl || 'N/A',
          execution_mode: executionMode,
          environment: environment,
          completed_at: new Date().toISOString()
        };
        
        results.push(result);
        
        if (testUrl) {
          cloudUrls.push(`${scenario.name}: ${testUrl}`);
        }
        
        // Store detailed output for analysis
        detailedMetrics.push({
          test: scenario.name,
          stdout_lines: stdout.split('\n').length,
          stderr_lines: stderr.split('\n').length,
          has_url: !!testUrl
        });
        
        resolve(result);
      } else {
        console.error(`❌ ${scenario.name} FAILED (exit code ${code})`);
        
        const result = {
          test: scenario.name,
          status: 'FAILED',
          error: `Exit code ${code}`,
          duration: `${duration}s`,
          vus: scenario.vus,
          execution_mode: executionMode,
          environment: environment,
          completed_at: new Date().toISOString()
        };
        
        results.push(result);
        reject(new Error(`Test failed with exit code ${code}`));
      }
    });
    
    // Handle spawn errors
    childProcess.on('error', (error) => {
      console.error(`❌ ${scenario.name} ERROR:`, error.message);
      
      results.push({
        test: scenario.name,
        status: 'ERROR',
        error: error.message,
        execution_mode: executionMode,
        environment: environment
      });
      
      reject(error);
    });
  });
}

// Main orchestration function
async function runOrchestrator() {
  const estimatedMinutes = scenarios.length * (testScale === 'small' ? 2 : 5);
  console.log(`\n🎯 Starting ${testScale} test suite on ${environment}`);
  console.log(`📈 Estimated time: ~${estimatedMinutes} minutes`);
  console.log(`🌩️ Execution: ${executionMode} mode\n`);
  
  const startTime = Date.now();
  let successCount = 0;
  let failureCount = 0;
  
  // Run tests sequentially
  for (let i = 0; i < scenarios.length; i++) {
    try {
      await runTest(scenarios[i], i + 1);
      successCount++;
      
      // Pause between tests (avoid overwhelming k6 cloud API)
      if (i < scenarios.length - 1) {
        const pauseSeconds = testScale === 'small' ? 10 : 30;
        console.log(`\n⏸️ Pausing ${pauseSeconds}s before next test...\n`);
        await new Promise(resolve => setTimeout(resolve, pauseSeconds * 1000));
      }
      
    } catch (error) {
      failureCount++;
      console.log(`💥 Continuing after failure...\n`);
      
      // Brief pause before continuing
      if (i < scenarios.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 15000));
      }
    }
  }
  
  const totalTime = Math.round((Date.now() - startTime) / 1000);
  await generateReports(successCount, failureCount, totalTime);
}

// Generate comprehensive reports
async function generateReports(successCount, failureCount, totalTime) {
  console.log('\n🎉 LOAD TEST ORCHESTRATOR COMPLETE\n');
  console.log('===================================\n');
  
  // Summary statistics
  const successRate = Math.round((successCount / scenarios.length) * 100);
  console.log('📊 EXECUTION SUMMARY:');
  console.log(`✅ Successful tests: ${successCount}/${scenarios.length} (${successRate}%)`);
  console.log(`❌ Failed tests: ${failureCount}/${scenarios.length}`);
  console.log(`⏱️ Total execution time: ${Math.round(totalTime / 60)} minutes`);
  console.log(`🌍 Environment: ${environment}`);
  console.log(`🚀 Mode: ${executionMode}`);
  
  // Generate CSV report
  const csvHeaders = 'Test Name,Status,VUs,Target Duration,Actual Duration,Environment,Mode,Test URL,Error,Completed At';
  const csvRows = results.map(r => 
    `"${r.test}","${r.status}",${r.vus},"${r.target_duration || 'N/A'}","${r.duration || 'N/A'}","${r.environment}","${r.execution_mode}","${r.url || 'N/A'}","${r.error || 'None'}","${r.completed_at || 'N/A'}"`
  );
  
  const csvContent = [csvHeaders, ...csvRows].join('\n');
  const csvFile = path.join(resultsDir, 'orchestrator_results.csv');
  fs.writeFileSync(csvFile, csvContent);
  console.log(`\n📁 CSV Report: ${csvFile}`);
  
  // Generate cloud URLs file
  if (executionMode === 'cloud' && cloudUrls.length > 0) {
    const urlsContent = [
      `# AutoGPT Platform Load Test URLs`,
      `# Environment: ${environment}`,
      `# Generated: ${new Date().toISOString()}`,
      `# Dashboard: https://significantgravitas.grafana.net/a/k6-app/`,
      '',
      ...cloudUrls,
      '',
      '# Direct Dashboard Access:',
      'https://significantgravitas.grafana.net/a/k6-app/'
    ].join('\n');
    
    const urlsFile = path.join(resultsDir, 'cloud_test_urls.txt');
    fs.writeFileSync(urlsFile, urlsContent);
    console.log(`📁 Cloud URLs: ${urlsFile}`);
  }
  
  // Generate detailed JSON report
  const jsonReport = {
    meta: {
      orchestrator_version: '1.0',
      environment: environment,
      execution_mode: executionMode,
      test_scale: testScale,
      total_scenarios: scenarios.length,
      generated_at: new Date().toISOString(),
      results_directory: resultsDir
    },
    summary: {
      successful_tests: successCount,
      failed_tests: failureCount,
      success_rate: `${successRate}%`,
      total_execution_time_seconds: totalTime,
      total_execution_time_minutes: Math.round(totalTime / 60)
    },
    test_results: results,
    detailed_metrics: detailedMetrics,
    cloud_urls: cloudUrls
  };
  
  const jsonFile = path.join(resultsDir, 'orchestrator_results.json');
  fs.writeFileSync(jsonFile, JSON.stringify(jsonReport, null, 2));
  console.log(`📁 JSON Report: ${jsonFile}`);
  
  // Display immediate results
  if (executionMode === 'cloud' && cloudUrls.length > 0) {
    console.log('\n🔗 K6 CLOUD TEST DASHBOARD URLS:');
    console.log('================================');
    cloudUrls.slice(0, 5).forEach(url => console.log(url));
    if (cloudUrls.length > 5) {
      console.log(`... and ${cloudUrls.length - 5} more URLs in ${urlsFile}`);
    }
    console.log('\n📈 Main Dashboard: https://significantgravitas.grafana.net/a/k6-app/');
  }
  
  console.log(`\n📂 All results saved in: ${resultsDir}/`);
  console.log('🏁 Load Test Orchestrator finished successfully!');
}

// Show usage help
function showUsage() {
  console.log('🎯 AutoGPT Platform Load Test Orchestrator\n');
  console.log('Usage: node load-test-orchestrator.js [ENVIRONMENT] [MODE] [SCALE]\n');
  console.log('ENVIRONMENT:');
  console.log('  LOCAL  - http://localhost:8006 (local development)');
  console.log('  DEV    - https://dev-api.agpt.co (development server)');
  console.log('  PROD   - https://api.agpt.co (production - coordinate with team!)\n');
  console.log('MODE:');
  console.log('  local  - Run locally with JSON output files');
  console.log('  cloud  - Run in k6 cloud with dashboard monitoring\n');
  console.log('SCALE:');
  console.log('  small  - 3 validation tests (~5 minutes)');
  console.log('  full   - 25 comprehensive tests (~2 hours)\n');
  console.log('Examples:');
  console.log('  node load-test-orchestrator.js DEV cloud small');
  console.log('  node load-test-orchestrator.js LOCAL local small');
  console.log('  node load-test-orchestrator.js DEV cloud full');
  console.log('  node load-test-orchestrator.js PROD cloud full  # Coordinate with team!\n');
  console.log('Requirements:');
  console.log('  - Pre-authenticated tokens generated (node generate-tokens.js)');
  console.log('  - k6 installed locally or run from Kubernetes pod');
  console.log('  - For cloud mode: K6_CLOUD_TOKEN and K6_CLOUD_PROJECT_ID set');
}

// Handle command line help
if (args.includes('--help') || args.includes('-h')) {
  showUsage();
  process.exit(0);
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Orchestrator interrupted by user');
  console.log('📊 Generating partial results...');
  generateReports(
    results.filter(r => r.status === 'SUCCESS').length, 
    results.filter(r => r.status === 'FAILED').length, 
    0
  ).then(() => {
    console.log('🏃‍♂️ Partial results saved');
    process.exit(0);
  });
});

// Start orchestrator
if (require.main === module) {
  runOrchestrator().catch(error => {
    console.error('💥 Orchestrator failed:', error);
    process.exit(1);
  });
}

module.exports = { runOrchestrator, testScenarios };