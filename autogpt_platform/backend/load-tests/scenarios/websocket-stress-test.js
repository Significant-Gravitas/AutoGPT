import ws from 'k6/ws';
import { check, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { getEnvironmentConfig, PERFORMANCE_CONFIG } from '../configs/environment.js';
import { authenticateUser, getRandomTestUser } from '../utils/auth.js';

const config = getEnvironmentConfig();

// Custom metrics
const wsConnectionErrors = new Rate('ws_connection_errors');
const wsMessageTime = new Trend('ws_message_response_time');
const wsConnections = new Counter('ws_connections_total');
const wsMessages = new Counter('ws_messages_total');

// Test configuration
export const options = {
  stages: [
    { duration: '1m', target: 20 },   // Ramp up to 20 concurrent WS connections
    { duration: '3m', target: 20 },   // Stay at 20 connections
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    ws_connection_errors: ['rate<0.1'],
    ws_message_response_time: ['p(95)<2000'],
    checks: ['rate>0.9'],
  },
  ext: {
    loadimpact: {
      projectID: __ENV.K6_CLOUD_PROJECT_ID,
      name: 'AutoGPT Platform WebSocket Stress Test',
    },
  },
};

// Global test state
let userAuth = null;

export function setup() {
  console.log('🔌 Setting up WebSocket stress test...');
  
  // Authenticate a test user for the test
  const testUser = getRandomTestUser();
  console.log(`Authenticating user: ${testUser.email}`);
  
  try {
    userAuth = authenticateUser(testUser);
    console.log('✅ Authentication successful');
    return { userAuth };
  } catch (error) {
    console.error('❌ Setup authentication failed:', error);
    throw error;
  }
}

export default function (data) {
  const { userAuth } = data;
  
  if (!userAuth || !userAuth.access_token) {
    console.error('❌ No valid authentication token available');
    wsConnectionErrors.add(1);
    return;
  }
  
  // WebSocket URL with authentication
  const wsUrl = `${config.WS_BASE_URL}/ws?token=${userAuth.access_token}`;
  
  wsStressTestScenario(wsUrl);
}

function wsStressTestScenario(wsUrl) {
  const startTime = Date.now();
  
  const response = ws.connect(wsUrl, {}, function (socket) {
    wsConnections.add(1);
    
    socket.on('open', function () {
      console.log('✅ WebSocket connection established');
      
      // Send various types of messages to test the WebSocket server
      testWebSocketMessaging(socket);
    });
    
    socket.on('message', function (message) {
      const responseTime = Date.now() - startTime;
      wsMessageTime.add(responseTime);
      wsMessages.add(1);
      
      try {
        const data = JSON.parse(message);
        
        check(data, {
          'WebSocket message - valid JSON': () => true,
          'WebSocket message - has type field': (msg) => msg.type !== undefined,
        });
        
        console.log(`📨 Received message: ${data.type}`);
        
        // Handle different message types
        handleWebSocketMessage(socket, data);
        
      } catch (error) {
        console.error('❌ Failed to parse WebSocket message:', error);
      }
    });
    
    socket.on('error', function (error) {
      console.error('❌ WebSocket error:', error);
      wsConnectionErrors.add(1);
    });
    
    socket.on('close', function () {
      console.log('🔒 WebSocket connection closed');
    });
    
    // Keep connection alive for test duration
    socket.setTimeout(function () {
      console.log('⏰ WebSocket test timeout, closing connection');
      socket.close();
    }, 30000); // 30 seconds per connection
  });
  
  check(response, {
    'WebSocket connection successful': (r) => r && r.status === 101,
  });
  
  if (!response || response.status !== 101) {
    wsConnectionErrors.add(1);
    console.error('❌ WebSocket connection failed:', response ? response.status : 'No response');
  }
}

function testWebSocketMessaging(socket) {
  // Test different types of WebSocket messages
  
  // 1. Subscribe to graph execution updates
  setTimeout(() => {
    const subscribeMessage = {
      type: 'subscribe',
      channel: 'graph_executions',
      user_id: 'test-user'
    };
    
    socket.send(JSON.stringify(subscribeMessage));
    console.log('📤 Sent subscription message');
  }, 1000);
  
  // 2. Send heartbeat/ping messages
  let pingCount = 0;
  const pingInterval = setInterval(() => {
    if (pingCount < 5) {
      const pingMessage = {
        type: 'ping',
        timestamp: Date.now()
      };
      
      socket.send(JSON.stringify(pingMessage));
      console.log('📤 Sent ping message');
      pingCount++;
    } else {
      clearInterval(pingInterval);
    }
  }, 2000);
  
  // 3. Request status updates
  setTimeout(() => {
    const statusMessage = {
      type: 'get_status',
      resource: 'executions'
    };
    
    socket.send(JSON.stringify(statusMessage));
    console.log('📤 Sent status request');
  }, 5000);
  
  // 4. Test execution monitoring
  setTimeout(() => {
    const monitorMessage = {
      type: 'monitor_execution',
      execution_id: 'test-execution-id'
    };
    
    socket.send(JSON.stringify(monitorMessage));
    console.log('📤 Sent execution monitor request');
  }, 7000);
}

function handleWebSocketMessage(socket, data) {
  switch (data.type) {
    case 'pong':
      check(data, {
        'Pong message - has timestamp': (msg) => msg.timestamp !== undefined,
      });
      break;
      
    case 'subscription_confirmed':
      check(data, {
        'Subscription confirmed - has channel': (msg) => msg.channel !== undefined,
      });
      break;
      
    case 'execution_update':
      check(data, {
        'Execution update - has execution_id': (msg) => msg.execution_id !== undefined,
        'Execution update - has status': (msg) => msg.status !== undefined,
      });
      break;
      
    case 'status_response':
      check(data, {
        'Status response - has data': (msg) => msg.data !== undefined,
      });
      break;
      
    case 'error':
      console.error('❌ WebSocket server error:', data.message);
      wsConnectionErrors.add(1);
      break;
      
    default:
      console.log(`📨 Unknown message type: ${data.type}`);
  }
}

export function teardown(data) {
  console.log('🧹 Cleaning up WebSocket stress test...');
  console.log(`Total WebSocket connections: ${wsConnections.value}`);
  console.log(`Total WebSocket messages: ${wsMessages.value}`);
}