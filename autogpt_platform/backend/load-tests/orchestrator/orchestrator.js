#!/usr/bin/env node

/**
 * AutoGPT Platform Load Test Orchestrator
 * 
 * Optimized test suite with only the highest VU count for each unique test type.
 * Eliminates duplicate tests and focuses on maximum load testing.
 */

import { spawn } from 'child_process';
import fs from 'fs';

console.log("🎯 AUTOGPT PLATFORM LOAD TEST ORCHESTRATOR\n");
console.log("===========================================\n");

// Parse command line arguments
const args = process.argv.slice(2);
const environment = args[0] || "DEV"; // LOCAL, DEV, PROD
const executionMode = args[1] || "cloud"; // local, cloud

console.log(`🌍 Target Environment: ${environment}`);
console.log(`🚀 Execution Mode: ${executionMode}`);

// Unified test scenarios - only highest VUs for each unique test
const unifiedTestScenarios = [
  // 1. Marketplace Public Access (highest VUs: 314)
  {
    name: "Marketplace_Public_Access_Max_Load",
    file: "tests/marketplace/public-access-test.js",
    vus: 314,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    description: "Public marketplace browsing at maximum load"
  },

  // 2. Marketplace Authenticated Access (highest VUs: 157) 
  {
    name: "Marketplace_Authenticated_Access_Max_Load",
    file: "tests/marketplace/library-access-test.js", 
    vus: 157,
    duration: "3m",
    rampUp: "30s", 
    rampDown: "30s",
    description: "Authenticated marketplace/library operations at maximum load"
  },

  // 3. Core API Load Test (highest VUs: 100)
  {
    name: "Core_API_Max_Load",
    file: "tests/api/core-api-test.js",
    vus: 100,
    duration: "5m",
    rampUp: "1m",
    rampDown: "1m", 
    description: "Core authenticated API endpoints at maximum load"
  },

  // 4. Graph Execution Load Test (highest VUs: 100)
  {
    name: "Graph_Execution_Max_Load", 
    file: "tests/api/graph-execution-test.js",
    vus: 100,
    duration: "5m",
    rampUp: "1m",
    rampDown: "1m",
    description: "Graph workflow execution pipeline at maximum load"
  },

  // 5. Credits API Single Endpoint (upgraded to 100 VUs)
  {
    name: "Credits_API_Max_Load",
    file: "tests/basic/single-endpoint-test.js",
    vus: 100,
    duration: "3m", 
    rampUp: "30s",
    rampDown: "30s",
    env: { ENDPOINT: "credits", CONCURRENT_REQUESTS: "1" },
    description: "Credits API endpoint at maximum load"
  },

  // 6. Graphs API Single Endpoint (upgraded to 100 VUs)
  {
    name: "Graphs_API_Max_Load",
    file: "tests/basic/single-endpoint-test.js", 
    vus: 100,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s", 
    env: { ENDPOINT: "graphs", CONCURRENT_REQUESTS: "1" },
    description: "Graphs API endpoint at maximum load"
  },

  // 7. Blocks API Single Endpoint (upgraded to 100 VUs)
  {
    name: "Blocks_API_Max_Load",
    file: "tests/basic/single-endpoint-test.js",
    vus: 100, 
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s",
    env: { ENDPOINT: "blocks", CONCURRENT_REQUESTS: "1" },
    description: "Blocks API endpoint at maximum load"
  },

  // 8. Executions API Single Endpoint (upgraded to 100 VUs) 
  {
    name: "Executions_API_Max_Load",
    file: "tests/basic/single-endpoint-test.js",
    vus: 100,
    duration: "3m",
    rampUp: "30s", 
    rampDown: "30s",
    env: { ENDPOINT: "executions", CONCURRENT_REQUESTS: "1" },
    description: "Executions API endpoint at maximum load"
  },

  // 9. Comprehensive Platform Journey (highest VUs: 100)
  {
    name: "Comprehensive_Platform_Max_Load",
    file: "tests/comprehensive/platform-journey-test.js",
    vus: 100,
    duration: "3m",
    rampUp: "30s",
    rampDown: "30s", 
    description: "End-to-end user journey simulation at maximum load"
  },

  // 10. Marketplace Stress Test (highest VUs: 500)
  {
    name: "Marketplace_Stress_Test",
    file: "tests/marketplace/public-access-test.js",
    vus: 500,
    duration: "2m",
    rampUp: "1m", 
    rampDown: "1m",
    description: "Ultimate marketplace stress test"
  },

  // 11. Core API Stress Test (highest VUs: 500)
  {
    name: "Core_API_Stress_Test", 
    file: "tests/api/core-api-test.js",
    vus: 500,
    duration: "2m",
    rampUp: "1m",
    rampDown: "1m",
    description: "Ultimate core API stress test"
  },

  // 12. Long Duration Core API Test (highest VUs: 100, longest duration)
  {
    name: "Long_Duration_Core_API_Test",
    file: "tests/api/core-api-test.js", 
    vus: 100,
    duration: "10m",
    rampUp: "1m",
    rampDown: "1m",
    description: "Extended duration core API endurance test"
  }
];

// Configuration
const K6_CLOUD_TOKEN = process.env.K6_CLOUD_TOKEN || '9347b8bd716cadc243e92f7d2f89107febfb81b49f2340d17da515d7b0513b51';
const K6_CLOUD_PROJECT_ID = process.env.K6_CLOUD_PROJECT_ID || '4254406';
const PAUSE_BETWEEN_TESTS = 30; // seconds

/**
 * Sleep for specified milliseconds
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Run a single k6 test
 */
async function runTest(test, index) {
  return new Promise((resolve, reject) => {
    console.log(`\n🚀 Test ${index + 1}/${unifiedTestScenarios.length}: ${test.name}`);
    console.log(`📊 Config: ${test.vus} VUs × ${test.duration} (${executionMode} mode)`);
    console.log(`📁 Script: ${test.file}`);
    console.log(`📋 Description: ${test.description}`);
    console.log(`⏱️ Test started: ${new Date().toISOString()}`);

    const env = {
      K6_CLOUD_TOKEN,
      K6_CLOUD_PROJECT_ID,
      K6_ENVIRONMENT: environment,
      VUS: test.vus.toString(),
      DURATION: test.duration,
      RAMP_UP: test.rampUp,
      RAMP_DOWN: test.rampDown,
      ...test.env
    };

    let args;
    if (executionMode === 'cloud') {
      args = [
        'cloud', 'run',
        ...Object.entries(env).map(([key, value]) => ['--env', `${key}=${value}`]).flat(),
        test.file
      ];
    } else {
      args = [
        'run',
        ...Object.entries(env).map(([key, value]) => ['--env', `${key}=${value}`]).flat(),
        test.file
      ];
    }

    const k6Process = spawn('k6', args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, ...env }
    });

    let output = '';
    let testId = null;

    k6Process.stdout.on('data', (data) => {
      const str = data.toString();
      output += str;
      
      // Extract test ID from k6 cloud output
      const testIdMatch = str.match(/Test created: .*\/(\d+)/);
      if (testIdMatch) {
        testId = testIdMatch[1];
        console.log(`🔗 Test URL: https://significantgravitas.grafana.net/a/k6-app/runs/${testId}`);
      }

      // Show progress updates
      const progressMatch = str.match(/(\d+)%/);
      if (progressMatch) {
        process.stdout.write(`\r⏳ Progress: ${progressMatch[1]}%`);
      }
    });

    k6Process.stderr.on('data', (data) => {
      output += data.toString();
    });

    k6Process.on('close', (code) => {
      process.stdout.write('\n'); // Clear progress line
      
      if (code === 0) {
        console.log(`✅ ${test.name} SUCCESS`);
        resolve({ 
          success: true, 
          testId, 
          url: testId ? `https://significantgravitas.grafana.net/a/k6-app/runs/${testId}` : 'unknown',
          vus: test.vus,
          duration: test.duration
        });
      } else {
        console.log(`❌ ${test.name} FAILED (exit code ${code})`);
        resolve({ 
          success: false, 
          testId, 
          url: testId ? `https://significantgravitas.grafana.net/a/k6-app/runs/${testId}` : 'unknown', 
          exitCode: code,
          vus: test.vus,
          duration: test.duration
        });
      }
    });

    k6Process.on('error', (error) => {
      console.log(`❌ ${test.name} ERROR: ${error.message}`);
      reject(error);
    });
  });
}

/**
 * Main execution
 */
async function main() {
  console.log(`\n📋 UNIFIED TEST PLAN`);
  console.log(`📊 Total tests: ${unifiedTestScenarios.length} (reduced from 25 original tests)`);
  console.log(`⏱️ Estimated duration: ~60 minutes\n`);

  console.log(`📋 Test Summary:`);
  unifiedTestScenarios.forEach((test, i) => {
    console.log(`  ${i + 1}. ${test.name} (${test.vus} VUs × ${test.duration})`);
  });
  console.log('');

  const results = [];

  for (let i = 0; i < unifiedTestScenarios.length; i++) {
    const test = unifiedTestScenarios[i];
    
    try {
      const result = await runTest(test, i);
      results.push({ ...test, ...result });
      
      // Pause between tests (except after the last one)
      if (i < unifiedTestScenarios.length - 1) {
        console.log(`\n⏸️ Pausing ${PAUSE_BETWEEN_TESTS}s before next test...`);
        await sleep(PAUSE_BETWEEN_TESTS * 1000);
      }
    } catch (error) {
      console.error(`💥 Fatal error running ${test.name}:`, error.message);
      results.push({ ...test, success: false, error: error.message });
    }
  }

  // Summary
  console.log('\n' + '='.repeat(60));
  console.log('🏁 UNIFIED LOAD TEST RESULTS SUMMARY');
  console.log('='.repeat(60));

  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);

  console.log(`✅ Successful tests: ${successful.length}/${results.length} (${Math.round(successful.length / results.length * 100)}%)`);
  console.log(`❌ Failed tests: ${failed.length}/${results.length}`);

  if (successful.length > 0) {
    console.log('\n✅ SUCCESSFUL TESTS:');
    successful.forEach(test => {
      console.log(`   • ${test.name} (${test.vus} VUs) - ${test.url}`);
    });
  }

  if (failed.length > 0) {
    console.log('\n❌ FAILED TESTS:');
    failed.forEach(test => {
      console.log(`   • ${test.name} (${test.vus} VUs) - ${test.url || 'no URL'} (exit: ${test.exitCode || 'unknown'})`);
    });
  }

  // Calculate total VU-minutes tested
  const totalVuMinutes = results.reduce((sum, test) => {
    const minutes = parseFloat(test.duration.replace(/[ms]/g, ''));
    const multiplier = test.duration.includes('m') ? 1 : (1/60); // convert seconds to minutes
    return sum + (test.vus * minutes * multiplier);
  }, 0);

  console.log(`\n📊 LOAD TESTING SUMMARY:`);
  console.log(`   • Total VU-minutes tested: ${Math.round(totalVuMinutes)}`);
  console.log(`   • Peak concurrent VUs: ${Math.max(...results.map(r => r.vus))}`);
  console.log(`   • Average test duration: ${(results.reduce((sum, r) => sum + parseFloat(r.duration.replace(/[ms]/g, '')), 0) / results.length).toFixed(1)}${results[0].duration.includes('m') ? 'm' : 's'}`);

  // Write results to file
  const timestamp = Math.floor(Date.now() / 1000);
  const resultsFile = `unified-results-${timestamp}.json`;
  fs.writeFileSync(resultsFile, JSON.stringify(results, null, 2));
  console.log(`\n📄 Detailed results saved to: ${resultsFile}`);

  console.log(`\n🎉 UNIFIED LOAD TEST ORCHESTRATOR COMPLETE\n`);

  process.exit(failed.length === 0 ? 0 : 1);
}

// Run if called directly
if (process.argv[1] === new URL(import.meta.url).pathname) {
  main().catch(error => {
    console.error('💥 Fatal error:', error);
    process.exit(1);
  });
}