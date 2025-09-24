#!/usr/bin/env node

/**
 * AutoGPT Platform Load Test Runner
 * Clean, automated test execution with result collection
 */

const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Test configurations
const TESTS = [
  {
    name: 'Core API Test',
    file: 'core-api-load-test.js',
    description: 'Tests authenticated API endpoints (credits, graphs, blocks)',
    local_config: { VUS: 5, DURATION: '30s', RAMP_UP: '10s', RAMP_DOWN: '10s' },
    cloud_config: { VUS: 50, DURATION: '3m', RAMP_UP: '30s', RAMP_DOWN: '30s' }
  },
  {
    name: 'Graph Execution Test',  
    file: 'graph-execution-load-test.js',
    description: 'Tests graph creation and execution workflows',
    local_config: { VUS: 3, DURATION: '30s', RAMP_UP: '10s', RAMP_DOWN: '10s' },
    cloud_config: { VUS: 20, DURATION: '3m', RAMP_UP: '30s', RAMP_DOWN: '30s' }
  },
  {
    name: 'Marketplace Access Test',
    file: 'marketplace-access-load-test.js', 
    description: 'Tests public marketplace browsing (no auth required)',
    local_config: { VUS: 10, DURATION: '30s', RAMP_UP: '10s', RAMP_DOWN: '10s' },
    cloud_config: { VUS: 100, DURATION: '3m', RAMP_UP: '30s', RAMP_DOWN: '30s' }
  }
];

const ENVIRONMENTS = {
  LOCAL: { K6_ENVIRONMENT: 'LOCAL' },
  DEV: { K6_ENVIRONMENT: 'DEV' },
  PROD: { K6_ENVIRONMENT: 'PROD' }
};

// Command line argument parsing
const args = process.argv.slice(2);
const command = args[0] || 'help';
const testName = args[1];
const environment = args[2] || 'DEV';
const mode = args[3] || 'local'; // local or cloud

function showHelp() {
  console.log(`
🚀 AutoGPT Platform Load Test Runner

Usage:
  node run-tests.js <command> [test] [environment] [mode]

Commands:
  list                          - List all available tests
  run <test> [env] [mode]      - Run specific test
  run-all [env] [mode]         - Run all tests
  verify                       - Run quick verification of all tests locally

Tests:
${TESTS.map(t => `  ${t.name.toLowerCase().replace(/ /g, '-').padEnd(25)} - ${t.description}`).join('\n')}

Environments:
  LOCAL                        - Local development server
  DEV                          - Development environment  
  PROD                         - Production environment

Modes:
  local                        - Run locally with low load
  cloud                        - Run on k6 cloud with high load

Examples:
  node run-tests.js verify
  node run-tests.js run core-api-test DEV local
  node run-tests.js run-all DEV cloud
  node run-tests.js list
`);
}

function listTests() {
  console.log('\n📋 Available Load Tests:\n');
  TESTS.forEach((test, index) => {
    console.log(`${index + 1}. ${test.name}`);
    console.log(`   File: ${test.file}`);
    console.log(`   Description: ${test.description}`);
    console.log(`   Local Config: ${JSON.stringify(test.local_config)}`);
    console.log(`   Cloud Config: ${JSON.stringify(test.cloud_config)}`);
    console.log('');
  });
}

function findTest(name) {
  const normalizedName = name.toLowerCase().replace(/-/g, ' ');
  return TESTS.find(test => 
    test.name.toLowerCase() === normalizedName ||
    test.file === name ||
    test.name.toLowerCase().includes(normalizedName)
  );
}

async function runTest(test, env, isCloud = false) {
  return new Promise((resolve) => {
    const config = isCloud ? test.cloud_config : test.local_config;
    const envVars = ENVIRONMENTS[env] || ENVIRONMENTS.DEV;
    
    // Prepare environment variables
    const testEnv = {
      ...process.env,
      ...envVars,
      ...config
    };

    // Add k6 cloud configuration if needed
    if (isCloud) {
      testEnv.K6_CLOUD_PROJECT_ID = process.env.K6_CLOUD_PROJECT_ID;
      testEnv.K6_CLOUD_TOKEN = process.env.K6_CLOUD_TOKEN;
    }
    
    console.log(`\n🚀 Running ${test.name}...`);
    console.log(`   Environment: ${env}`);
    console.log(`   Mode: ${isCloud ? 'Cloud' : 'Local'}`);
    console.log(`   Config: ${JSON.stringify(config)}`);
    
    // Build k6 command
    const k6Args = ['run', test.file];
    if (isCloud) {
      k6Args.push('--out', 'cloud');
    } else {
      // Create results directory
      const resultsDir = path.join(process.cwd(), 'results');
      if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true });
      }
      
      const timestamp = new Date().toISOString().replace(/:/g, '-').slice(0, -5);
      const resultFile = path.join(resultsDir, `${test.name.toLowerCase().replace(/ /g, '-')}-${timestamp}.json`);
      const summaryFile = path.join(resultsDir, `${test.name.toLowerCase().replace(/ /g, '-')}-${timestamp}-summary.json`);
      
      k6Args.push('--out', `json=${resultFile}`);
      k6Args.push('--summary-export', summaryFile);
      k6Args.push('--quiet');
    }
    
    const k6Process = spawn('k6', k6Args, {
      env: testEnv,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let output = '';
    let errorOutput = '';
    
    k6Process.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      if (!isCloud) process.stdout.write(text);
    });
    
    k6Process.stderr.on('data', (data) => {
      const text = data.toString();
      errorOutput += text;
      if (!isCloud) process.stderr.write(text);
    });
    
    k6Process.on('close', (code) => {
      if (code === 0) {
        console.log(`✅ ${test.name} completed successfully`);
        if (isCloud && output.includes('https://app.k6.io/runs/')) {
          const match = output.match(/https:\/\/app\.k6\.io\/runs\/\d+/);
          if (match) {
            console.log(`🔗 Cloud Results: ${match[0]}`);
            
            // Save cloud URL to file
            const cloudResultsFile = path.join(process.cwd(), 'k6-cloud-results.txt');
            const timestamp = new Date().toISOString();
            const logEntry = `${timestamp} - ${test.name}: ${match[0]}\n`;
            fs.appendFileSync(cloudResultsFile, logEntry);
          }
        }
      } else {
        console.log(`❌ ${test.name} failed with code ${code}`);
        if (errorOutput) {
          console.log(`Error output: ${errorOutput}`);
        }
      }
      resolve(code === 0);
    });
  });
}

async function verifyTests() {
  console.log('\n🔍 Running quick verification of all tests...\n');
  
  let allPassed = true;
  for (const test of TESTS) {
    // Use minimal config for verification
    const quickTest = {
      ...test,
      local_config: { VUS: 1, DURATION: '10s', RAMP_UP: '0s', RAMP_DOWN: '0s' }
    };
    
    const passed = await runTest(quickTest, 'DEV', false);
    if (!passed) {
      allPassed = false;
      console.log(`❌ Verification failed for ${test.name}`);
    } else {
      console.log(`✅ Verification passed for ${test.name}`);
    }
  }
  
  console.log(`\n${allPassed ? '✅ All tests verified successfully' : '❌ Some tests failed verification'}`);
  return allPassed;
}

async function runAllTests(env, isCloud) {
  console.log(`\n🚀 Running all tests in ${env} environment (${isCloud ? 'cloud' : 'local'} mode)...\n`);
  
  const results = [];
  for (const test of TESTS) {
    const passed = await runTest(test, env, isCloud);
    results.push({ test: test.name, passed });
  }
  
  console.log('\n📊 Summary:');
  results.forEach(result => {
    console.log(`  ${result.passed ? '✅' : '❌'} ${result.test}`);
  });
  
  const totalPassed = results.filter(r => r.passed).length;
  console.log(`\n${totalPassed}/${results.length} tests passed`);
  
  return results;
}

// Main execution
async function main() {
  switch (command) {
    case 'help':
    case '--help':
    case '-h':
      showHelp();
      break;
      
    case 'list':
      listTests();
      break;
      
    case 'verify':
      await verifyTests();
      break;
      
    case 'run':
      if (!testName) {
        console.log('❌ Test name required. Use "node run-tests.js list" to see available tests.');
        process.exit(1);
      }
      
      const test = findTest(testName);
      if (!test) {
        console.log(`❌ Test "${testName}" not found. Use "node run-tests.js list" to see available tests.`);
        process.exit(1);
      }
      
      const isCloud = mode === 'cloud';
      await runTest(test, environment.toUpperCase(), isCloud);
      break;
      
    case 'run-all':
      const isCloudMode = environment === 'cloud' || mode === 'cloud';
      const targetEnv = isCloudMode ? (environment === 'cloud' ? 'DEV' : environment) : environment;
      await runAllTests(targetEnv.toUpperCase(), isCloudMode);
      break;
      
    default:
      console.log(`❌ Unknown command: ${command}`);
      showHelp();
      process.exit(1);
  }
}

main().catch(console.error);