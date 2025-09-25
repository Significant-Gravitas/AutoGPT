#!/usr/bin/env node

/**
 * Interactive Load Testing CLI Tool for AutoGPT Platform
 * 
 * This tool provides an interactive interface for running various load tests
 * against AutoGPT Platform APIs with customizable parameters.
 * 
 * Usage: node interactive-test.js
 */

import { execSync } from 'child_process';
import readline from 'readline';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Color utilities for better CLI experience
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m'
};

function colorize(text, color) {
  return `${colors[color]}${text}${colors.reset}`;
}

// Available test configurations
const TEST_CONFIGS = {
  'basic-connectivity': {
    name: 'Basic Connectivity Test',
    description: 'Tests basic health check + authentication endpoints',
    file: 'basic-connectivity-test.js',
    defaultVUs: 10,
    defaultDuration: '30s',
    maxVUs: 100,
    endpoints: ['health', 'auth']
  },
  'core-api': {
    name: 'Core API Load Test', 
    description: 'Tests main API endpoints: credits, graphs, blocks',
    file: 'core-api-load-test.js',
    defaultVUs: 10,
    defaultDuration: '30s',
    maxVUs: 50,
    endpoints: ['credits', 'graphs', 'blocks']
  },
  'comprehensive-platform': {
    name: 'Comprehensive Platform Test',
    description: 'Realistic user workflows across all platform features',
    file: 'scenarios/comprehensive-platform-load-test.js',
    defaultVUs: 5,
    defaultDuration: '30s',
    maxVUs: 20,
    endpoints: ['credits', 'graphs', 'blocks', 'executions']
  },
  'single-endpoint': {
    name: 'Single Endpoint Test',
    description: 'Test specific API endpoint with custom parameters',
    file: 'single-endpoint-test.js',
    defaultVUs: 3,
    defaultDuration: '20s',
    maxVUs: 10,
    endpoints: ['credits', 'graphs', 'blocks', 'executions'],
    requiresEndpoint: true
  }
};

// Environment configurations
const ENVIRONMENTS = {
  'local': {
    name: 'Local Development',
    description: 'http://localhost:8006',
    env: 'LOCAL'
  },
  'dev': {
    name: 'Development Server',
    description: 'https://dev-server.agpt.co',
    env: 'DEV'
  },
  'prod': {
    name: 'Production Server',
    description: 'https://api.agpt.co',
    env: 'PROD'
  }
};

class InteractiveLoadTester {
  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
  }

  async prompt(question) {
    return new Promise((resolve) => {
      this.rl.question(question, resolve);
    });
  }

  async run() {
    console.log(colorize('🚀 AutoGPT Platform Load Testing CLI', 'cyan'));
    console.log(colorize('=====================================', 'cyan'));
    console.log();

    try {
      // Step 1: Select test type
      const testType = await this.selectTestType();
      const testConfig = TEST_CONFIGS[testType];

      // Step 2: Select environment
      const environment = await this.selectEnvironment();

      // Step 3: Select execution mode (local vs cloud)
      const isCloud = await this.selectExecutionMode();

      // Step 4: Get test parameters
      const params = await this.getTestParameters(testConfig);

      // Step 5: Get endpoint for single endpoint test
      let endpoint = null;
      if (testConfig.requiresEndpoint) {
        endpoint = await this.selectEndpoint(testConfig.endpoints);
      }

      // Step 6: Execute the test
      await this.executeTest({
        testType,
        testConfig,
        environment,
        isCloud,
        params,
        endpoint
      });

    } catch (error) {
      console.error(colorize(`❌ Error: ${error.message}`, 'red'));
    } finally {
      this.rl.close();
    }
  }

  async selectTestType() {
    console.log(colorize('📋 Available Load Tests:', 'yellow'));
    console.log();

    Object.entries(TEST_CONFIGS).forEach(([key, config], index) => {
      console.log(colorize(`${index + 1}. ${config.name}`, 'green'));
      console.log(colorize(`   ${config.description}`, 'dim'));
      console.log(colorize(`   Endpoints: ${config.endpoints.join(', ')}`, 'dim'));
      console.log(colorize(`   Recommended: ${config.defaultVUs} VUs, ${config.defaultDuration}`, 'dim'));
      console.log();
    });

    while (true) {
      const choice = await this.prompt(colorize('Select test type (1-4): ', 'bright'));
      const index = parseInt(choice) - 1;
      const keys = Object.keys(TEST_CONFIGS);

      if (index >= 0 && index < keys.length) {
        return keys[index];
      }
      console.log(colorize('❌ Invalid choice. Please enter 1-4.', 'red'));
    }
  }

  async selectEnvironment() {
    console.log(colorize('🌍 Target Environment:', 'yellow'));
    console.log();

    Object.entries(ENVIRONMENTS).forEach(([key, config], index) => {
      console.log(colorize(`${index + 1}. ${config.name}`, 'green'));
      console.log(colorize(`   ${config.description}`, 'dim'));
      console.log();
    });

    while (true) {
      const choice = await this.prompt(colorize('Select environment (1-3): ', 'bright'));
      const index = parseInt(choice) - 1;
      const keys = Object.keys(ENVIRONMENTS);

      if (index >= 0 && index < keys.length) {
        return ENVIRONMENTS[keys[index]];
      }
      console.log(colorize('❌ Invalid choice. Please enter 1-3.', 'red'));
    }
  }

  async selectExecutionMode() {
    console.log(colorize('☁️  Execution Mode:', 'yellow'));
    console.log();
    console.log(colorize('1. Local Execution', 'green'));
    console.log(colorize('   Run test locally, results in terminal', 'dim'));
    console.log();
    console.log(colorize('2. k6 Cloud Execution', 'green'));
    console.log(colorize('   Run test on k6 cloud, get shareable results link', 'dim'));
    console.log();

    while (true) {
      const choice = await this.prompt(colorize('Select execution mode (1-2): ', 'bright'));
      
      if (choice === '1') {
        return false; // Local
      } else if (choice === '2') {
        return true; // Cloud
      }
      console.log(colorize('❌ Invalid choice. Please enter 1 or 2.', 'red'));
    }
  }

  async getTestParameters(testConfig) {
    console.log(colorize('⚙️  Test Parameters:', 'yellow'));
    console.log();

    // Get VUs
    const vusPrompt = colorize(`Virtual Users (1-${testConfig.maxVUs}) [${testConfig.defaultVUs}]: `, 'bright');
    const vusInput = await this.prompt(vusPrompt);
    const vus = parseInt(vusInput) || testConfig.defaultVUs;

    if (vus < 1 || vus > testConfig.maxVUs) {
      throw new Error(`VUs must be between 1 and ${testConfig.maxVUs}`);
    }

    // Get duration
    const durationPrompt = colorize(`Test duration (e.g., 30s, 2m) [${testConfig.defaultDuration}]: `, 'bright');
    const durationInput = await this.prompt(durationPrompt);
    const duration = durationInput || testConfig.defaultDuration;

    // Validate duration format
    if (!/^\d+[smh]$/.test(duration)) {
      throw new Error('Duration must be in format like 30s, 2m, 1h');
    }

    // Get requests per VU for applicable tests
    let requestsPerVU = 1;
    if (['core-api', 'comprehensive-platform'].includes(testConfig.file.replace('.js', '').replace('scenarios/', ''))) {
      const rpsPrompt = colorize('Requests per VU per iteration [1]: ', 'bright');
      const rpsInput = await this.prompt(rpsPrompt);
      requestsPerVU = parseInt(rpsInput) || 1;

      if (requestsPerVU < 1 || requestsPerVU > 50) {
        throw new Error('Requests per VU must be between 1 and 50');
      }
    }

    // Get concurrent requests for single endpoint test
    let concurrentRequests = 1;
    if (testConfig.requiresEndpoint) {
      const concurrentPrompt = colorize('Concurrent requests per VU per iteration [1]: ', 'bright');
      const concurrentInput = await this.prompt(concurrentPrompt);
      concurrentRequests = parseInt(concurrentInput) || 1;

      if (concurrentRequests < 1 || concurrentRequests > 500) {
        throw new Error('Concurrent requests must be between 1 and 500');
      }
    }

    return { vus, duration, requestsPerVU, concurrentRequests };
  }

  async selectEndpoint(endpoints) {
    console.log(colorize('🎯 Target Endpoint:', 'yellow'));
    console.log();

    endpoints.forEach((endpoint, index) => {
      console.log(colorize(`${index + 1}. /api/${endpoint}`, 'green'));
    });
    console.log();

    while (true) {
      const choice = await this.prompt(colorize(`Select endpoint (1-${endpoints.length}): `, 'bright'));
      const index = parseInt(choice) - 1;

      if (index >= 0 && index < endpoints.length) {
        return endpoints[index];
      }
      console.log(colorize(`❌ Invalid choice. Please enter 1-${endpoints.length}.`, 'red'));
    }
  }

  async executeTest({ testType, testConfig, environment, isCloud, params, endpoint }) {
    console.log();
    console.log(colorize('🚀 Executing Load Test...', 'magenta'));
    console.log(colorize('========================', 'magenta'));
    console.log();
    console.log(colorize(`Test: ${testConfig.name}`, 'bright'));
    console.log(colorize(`Environment: ${environment.name} (${environment.description})`, 'bright'));
    console.log(colorize(`Mode: ${isCloud ? 'k6 Cloud' : 'Local'}`, 'bright'));
    console.log(colorize(`VUs: ${params.vus}`, 'bright'));
    console.log(colorize(`Duration: ${params.duration}`, 'bright'));
    if (endpoint) {
      console.log(colorize(`Endpoint: /api/${endpoint}`, 'bright'));
      if (params.concurrentRequests > 1) {
        console.log(colorize(`Concurrent Requests: ${params.concurrentRequests} per VU`, 'bright'));
      }
    }
    console.log();

    // Build k6 command
    let command = 'k6 run';
    
    // Environment variables
    const envVars = [
      `K6_ENVIRONMENT=${environment.env}`,
      `VUS=${params.vus}`,
      `DURATION=${params.duration}`
    ];

    if (params.requestsPerVU > 1) {
      envVars.push(`REQUESTS_PER_VU=${params.requestsPerVU}`);
    }

    if (endpoint) {
      envVars.push(`ENDPOINT=${endpoint}`);
    }

    if (params.concurrentRequests > 1) {
      envVars.push(`CONCURRENT_REQUESTS=${params.concurrentRequests}`);
    }

    // Add cloud configuration if needed
    if (isCloud) {
      const cloudToken = process.env.K6_CLOUD_TOKEN;
      const cloudProjectId = process.env.K6_CLOUD_PROJECT_ID;
      
      if (!cloudToken || !cloudProjectId) {
        console.log(colorize('⚠️  k6 Cloud credentials not found in environment variables:', 'yellow'));
        console.log(colorize('   K6_CLOUD_TOKEN=your_token', 'dim'));
        console.log(colorize('   K6_CLOUD_PROJECT_ID=your_project_id', 'dim'));
        console.log();
        
        const proceed = await this.prompt(colorize('Continue with local execution instead? (y/n): ', 'bright'));
        if (proceed.toLowerCase() !== 'y') {
          throw new Error('k6 Cloud execution cancelled');
        }
        isCloud = false;
      } else {
        envVars.push(`K6_CLOUD_TOKEN=${cloudToken}`);
        envVars.push(`K6_CLOUD_PROJECT_ID=${cloudProjectId}`);
        command += ' --out cloud';
      }
    }

    // Build full command
    const fullCommand = `cd ${__dirname} && ${envVars.join(' ')} ${command} ${testConfig.file}`;

    console.log(colorize('Executing command:', 'dim'));
    console.log(colorize(fullCommand, 'dim'));
    console.log();

    try {
      const result = execSync(fullCommand, { 
        stdio: 'inherit',
        maxBuffer: 1024 * 1024 * 10 // 10MB buffer
      });

      console.log();
      console.log(colorize('✅ Test completed successfully!', 'green'));
      
      if (isCloud) {
        console.log();
        console.log(colorize('🌐 Check your k6 Cloud dashboard for detailed results:', 'cyan'));
        console.log(colorize('   https://app.k6.io/dashboard', 'cyan'));
      }

    } catch (error) {
      console.log();
      console.log(colorize('❌ Test execution failed:', 'red'));
      console.log(colorize(error.message, 'red'));
      
      if (error.status) {
        console.log(colorize(`Exit code: ${error.status}`, 'dim'));
      }
    }
  }
}

// Run the interactive tool
if (import.meta.url === `file://${process.argv[1]}`) {
  const tester = new InteractiveLoadTester();
  tester.run().catch(console.error);
}

export default InteractiveLoadTester;