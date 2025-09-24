#!/usr/bin/env node

/**
 * Automated High-Concurrency Cloud Test Runner
 * Runs all tests on k6 cloud with high load configurations
 */

const { spawn } = require('child_process');
const fs = require('fs');

// High-concurrency test configurations based on performance analysis
const CLOUD_TESTS = [
  {
    name: 'Core API High Load Test',
    file: 'core-api-load-test.js',
    description: 'Test authenticated API endpoints at high concurrency',
    config: {
      VUS: 80,              // Target ~80 RPS (within DB connection limits)
      DURATION: '5m',       // 5 minutes sustained load
      RAMP_UP: '1m',        // Gradual ramp up
      RAMP_DOWN: '1m'       // Gradual ramp down
    }
  },
  {
    name: 'Graph Execution Stress Test',
    file: 'graph-execution-load-test.js', 
    description: 'Test graph creation/execution pipeline under stress',
    config: {
      VUS: 30,              // Lower due to complex operations
      DURATION: '5m',       // 5 minutes sustained load
      RAMP_UP: '1m',
      RAMP_DOWN: '1m',
      REQUESTS_PER_VU: 2    // 2 concurrent ops per VU
    }
  },
  {
    name: 'Marketplace Scale Test',
    file: 'marketplace-access-load-test.js',
    description: 'Test public marketplace at scale (no auth bottleneck)',
    config: {
      VUS: 150,             // Higher load for public endpoints
      DURATION: '5m',       // 5 minutes sustained load  
      RAMP_UP: '1m',
      RAMP_DOWN: '1m'
    }
  }
];

// Load k6 credentials from file or environment
function loadK6Credentials() {
  const credentialsFile = require('path').join(__dirname, 'configs', 'k6-credentials.env');
  
  let credentials = {
    K6_CLOUD_TOKEN: process.env.K6_CLOUD_TOKEN,
    K6_CLOUD_PROJECT_ID: process.env.K6_CLOUD_PROJECT_ID
  };
  
  // Try to load from credentials file
  if (require('fs').existsSync(credentialsFile)) {
    const envContent = require('fs').readFileSync(credentialsFile, 'utf8');
    const envLines = envContent.split('\n').filter(line => line.trim() && !line.startsWith('#'));
    
    envLines.forEach(line => {
      const equalIndex = line.indexOf('=');
      if (equalIndex > 0) {
        const key = line.substring(0, equalIndex).trim();
        const value = line.substring(equalIndex + 1).trim();
        if (key && value) {
          credentials[key] = value;
        }
      }
    });
  }
  
  return credentials;
}

const REQUIRED_ENV_VARS = loadK6Credentials();

function checkEnvironment() {
  console.log('🔍 Checking environment variables...');
  
  const missing = [];
  for (const [key, value] of Object.entries(REQUIRED_ENV_VARS)) {
    if (!value) {
      missing.push(key);
    }
  }
  
  if (missing.length > 0) {
    console.error(`❌ Missing required k6 cloud credentials: ${missing.join(', ')}`);
    console.error('\nPlease either:');
    console.error('1. Copy configs/k6-credentials.env.example to configs/k6-credentials.env and fill in your credentials');
    console.error('2. Or set environment variables:');
    console.error('   export K6_CLOUD_TOKEN="your-k6-cloud-token"');
    console.error('   export K6_CLOUD_PROJECT_ID="your-project-id"');
    console.error('\nGet credentials from: https://app.k6.io/');
    process.exit(1);
  }
  
  console.log('✅ Environment variables configured');
}

async function runCloudTest(test) {
  return new Promise((resolve) => {
    console.log(`\n🚀 Starting ${test.name}...`);
    console.log(`   File: ${test.file}`);
    console.log(`   Description: ${test.description}`);
    console.log(`   Config: ${JSON.stringify(test.config)}`);
    
    // Prepare environment variables
    const testEnv = {
      ...process.env,
      K6_ENVIRONMENT: 'DEV',
      ...test.config,
      ...REQUIRED_ENV_VARS
    };
    
    const k6Args = ['run', test.file, '--out', 'cloud'];
    
    const k6Process = spawn('k6', k6Args, {
      env: testEnv,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    let output = '';
    let errorOutput = '';
    
    k6Process.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      // Show real-time output for cloud tests
      process.stdout.write(text);
    });
    
    k6Process.stderr.on('data', (data) => {
      const text = data.toString();
      errorOutput += text;
      process.stderr.write(text);
    });
    
    k6Process.on('close', (code) => {
      const result = {
        test: test.name,
        success: code === 0,
        config: test.config,
        cloudUrl: null
      };
      
      if (code === 0) {
        console.log(`✅ ${test.name} started successfully on k6 cloud`);
        
        // Extract cloud URL
        const match = output.match(/https:\/\/app\.k6\.io\/runs\/\d+/);
        if (match) {
          result.cloudUrl = match[0];
          console.log(`🔗 Cloud Results: ${result.cloudUrl}`);
          
          // Save to results file
          const timestamp = new Date().toISOString();
          const logEntry = `${timestamp} - ${test.name}: ${result.cloudUrl}\n`;
          fs.appendFileSync('k6-cloud-results.txt', logEntry);
        }
      } else {
        console.log(`❌ ${test.name} failed to start (code ${code})`);
        if (errorOutput) {
          console.log(`Error: ${errorOutput}`);
        }
      }
      
      resolve(result);
    });
  });
}

async function runAllCloudTests() {
  console.log('🌩️  AutoGPT Platform High-Concurrency Cloud Load Tests');
  console.log('=======================================================');
  console.log(`Running ${CLOUD_TESTS.length} tests on k6 cloud with high-load configurations\n`);
  
  checkEnvironment();
  
  // Create results file header
  const timestamp = new Date().toISOString();
  const header = `\\n=== High-Concurrency Cloud Test Session - ${timestamp} ===\\n`;
  fs.appendFileSync('k6-cloud-results.txt', header);
  
  const results = [];
  
  for (const test of CLOUD_TESTS) {
    const result = await runCloudTest(test);
    results.push(result);
    
    // Wait between tests to avoid overwhelming k6 cloud
    if (test !== CLOUD_TESTS[CLOUD_TESTS.length - 1]) {
      console.log('⏳ Waiting 30 seconds before next test...');
      await new Promise(resolve => setTimeout(resolve, 30000));
    }
  }
  
  // Summary
  console.log('\\n📊 High-Concurrency Cloud Test Results:');
  console.log('==========================================');
  
  results.forEach(result => {
    const status = result.success ? '✅' : '❌';
    const url = result.cloudUrl ? `\\n   🔗 ${result.cloudUrl}` : '';
    const config = `\\n   ⚙️  ${JSON.stringify(result.config)}`;
    console.log(`${status} ${result.test}${config}${url}`);
  });
  
  const successful = results.filter(r => r.success).length;
  const totalLoad = results.reduce((sum, r) => sum + (r.config.VUS || 0), 0);
  
  console.log(`\\n✨ Summary: ${successful}/${results.length} tests started successfully`);
  console.log(`🚀 Total Virtual Users: ${totalLoad} VUs across all tests`);
  console.log(`📈 Expected Combined RPS: ~${Math.floor(totalLoad * 0.8)} RPS`);
  console.log('\\n📋 All cloud test URLs saved to: k6-cloud-results.txt');
  
  if (successful === results.length) {
    console.log('\\n🎉 All high-concurrency tests are now running on k6 cloud!');
    console.log('Monitor results at: https://app.k6.io/');
  } else {
    console.log('\\n⚠️  Some tests failed to start. Check error messages above.');
  }
  
  return results;
}

// Main execution
if (require.main === module) {
  runAllCloudTests().catch(console.error);
}

module.exports = { runAllCloudTests, CLOUD_TESTS };