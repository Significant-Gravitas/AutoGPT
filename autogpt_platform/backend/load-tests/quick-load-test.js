// Quick version of the main load test to identify failures
import { options as mainOptions, default as mainScenario } from './scenarios/load-test.js';

// Reduced load configuration for debugging
export const options = {
  stages: [
    { duration: '15s', target: 2 }, // Much lower concurrency
  ],
  thresholds: {
    http_req_failed: ['rate<0.1'], // Allow 10% failures for debugging
    http_req_duration: ['p(95)<10000'], // 10s timeout
  }
};

export default mainScenario;