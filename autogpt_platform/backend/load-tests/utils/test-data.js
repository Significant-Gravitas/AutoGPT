/**
 * Test data generators for AutoGPT Platform load tests
 */

/**
 * Generate sample graph data for testing
 */
export function generateTestGraph(name = null) {
  const graphName = name || `Load Test Graph ${Math.random().toString(36).substr(2, 9)}`;
  
  return {
    name: graphName,
    description: "Generated graph for load testing purposes",
    graph: {
      name: graphName,
      description: "Load testing graph",
      nodes: [
        {
          id: "input_node",
          name: "Agent Input",
          block_id: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b", // AgentInputBlock ID
          input_default: {
            name: "Load Test Input",
            description: "Test input for load testing",
            placeholder_values: {}
          },
          input_nodes: [],
          output_nodes: ["output_node"],
          metadata: {
            position: { x: 100, y: 100 }
          }
        },
        {
          id: "output_node",
          name: "Agent Output", 
          block_id: "363ae599-353e-4804-937e-b2ee3cef3da4", // AgentOutputBlock ID
          input_default: {
            name: "Load Test Output",
            description: "Test output for load testing",
            value: "Test output value"
          },
          input_nodes: ["input_node"],
          output_nodes: [],
          metadata: {
            position: { x: 300, y: 100 }
          }
        }
      ],
      links: [
        {
          source_id: "input_node",
          sink_id: "output_node",
          source_name: "result",
          sink_name: "value"
        }
      ]
    }
  };
}

/**
 * Generate test execution inputs for graph execution
 */
export function generateExecutionInputs() {
  return {
    "Load Test Input": {
      name: "Load Test Input",
      description: "Test input for load testing",
      placeholder_values: {
        test_data: `Test execution at ${new Date().toISOString()}`,
        test_parameter: Math.random().toString(36).substr(2, 9),
        numeric_value: Math.floor(Math.random() * 1000)
      }
    }
  };
}

/**
 * Generate a more complex graph for execution testing
 */
export function generateComplexTestGraph(name = null) {
  const graphName = name || `Complex Load Test Graph ${Math.random().toString(36).substr(2, 9)}`;
  
  return {
    name: graphName,
    description: "Complex graph for load testing with multiple blocks",
    graph: {
      name: graphName,
      description: "Multi-block load testing graph",
      nodes: [
        {
          id: "input_node",
          name: "Agent Input",
          block_id: "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b", // AgentInputBlock ID
          input_default: {
            name: "Load Test Input",
            description: "Test input for load testing",
            placeholder_values: {}
          },
          input_nodes: [],
          output_nodes: ["time_node"],
          metadata: {
            position: { x: 100, y: 100 }
          }
        },
        {
          id: "time_node", 
          name: "Get Current Time",
          block_id: "a892b8d9-3e4e-4e9c-9c1e-75f8efcf1bfa", // GetCurrentTimeBlock ID
          input_default: {
            trigger: "test",
            format_type: {
              discriminator: "iso8601",
              timezone: "UTC"
            }
          },
          input_nodes: ["input_node"],
          output_nodes: ["output_node"],
          metadata: {
            position: { x: 250, y: 100 }
          }
        },
        {
          id: "output_node",
          name: "Agent Output", 
          block_id: "363ae599-353e-4804-937e-b2ee3cef3da4", // AgentOutputBlock ID
          input_default: {
            name: "Load Test Output",
            description: "Test output for load testing",
            value: "Test output value"
          },
          input_nodes: ["time_node"],
          output_nodes: [],
          metadata: {
            position: { x: 400, y: 100 }
          }
        }
      ],
      links: [
        {
          source_id: "input_node",
          sink_id: "time_node",
          source_name: "result",
          sink_name: "trigger"
        },
        {
          source_id: "time_node",
          sink_id: "output_node", 
          source_name: "time",
          sink_name: "value"
        }
      ]
    }
  };
}

/**
 * Generate test file content for upload testing
 */
export function generateTestFileContent(sizeKB = 10) {
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  const targetLength = sizeKB * 1024;
  let content = '';
  
  for (let i = 0; i < targetLength; i++) {
    content += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  
  return content;
}

/**
 * Generate schedule data for testing
 */
export function generateScheduleData(graphId) {
  return {
    name: `Load Test Schedule ${Math.random().toString(36).substr(2, 9)}`,
    cron: "*/5 * * * *", // Every 5 minutes
    inputs: generateExecutionInputs(),
    credentials: {},
    timezone: "UTC"
  };
}

/**
 * Generate API key creation request
 */
export function generateAPIKeyRequest() {
  return {
    name: `Load Test API Key ${Math.random().toString(36).substr(2, 9)}`,
    description: "Generated for load testing",
    permissions: ["read", "write", "execute"]
  };
}

/**
 * Generate credit top-up request
 */
export function generateTopUpRequest() {
  return {
    credit_amount: Math.floor(Math.random() * 1000) + 100 // 100-1100 credits
  };
}

/**
 * Generate notification preferences
 */
export function generateNotificationPreferences() {
  return {
    email_notifications: Math.random() > 0.5,
    webhook_notifications: Math.random() > 0.5,
    notification_frequency: ["immediate", "daily", "weekly"][Math.floor(Math.random() * 3)]
  };
}

/**
 * Generate block execution data
 */
export function generateBlockExecutionData(blockId) {
  const commonInputs = {
    GetCurrentTimeBlock: {
      trigger: "test",
      format_type: {
        discriminator: "iso8601",
        timezone: "UTC"
      }
    },
    HttpRequestBlock: {
      url: "https://httpbin.org/get",
      method: "GET",
      headers: {}
    },
    TextProcessorBlock: {
      text: `Load test input ${Math.random().toString(36).substr(2, 9)}`,
      operation: "uppercase"
    },
    CalculatorBlock: {
      expression: `${Math.floor(Math.random() * 100)} + ${Math.floor(Math.random() * 100)}`
    }
  };
  
  return commonInputs[blockId] || {
    generic_input: `Test data for ${blockId}`,
    test_id: Math.random().toString(36).substr(2, 9)
  };
}

/**
 * Generate realistic user onboarding data
 */
export function generateOnboardingData() {
  return {
    completed_steps: ["welcome", "first_graph"],
    current_step: "explore_blocks",
    preferences: {
      use_case: ["automation", "data_processing", "integration"][Math.floor(Math.random() * 3)],
      experience_level: ["beginner", "intermediate", "advanced"][Math.floor(Math.random() * 3)]
    }
  };
}

/**
 * Generate realistic integration credentials
 */
export function generateIntegrationCredentials(provider) {
  const templates = {
    github: {
      access_token: `ghp_${Math.random().toString(36).substr(2, 36)}`,
      scope: "repo,user"
    },
    google: {
      access_token: `ya29.${Math.random().toString(36).substr(2, 100)}`,
      refresh_token: `1//${Math.random().toString(36).substr(2, 50)}`,
      scope: "https://www.googleapis.com/auth/gmail.readonly"
    },
    slack: {
      access_token: `xoxb-${Math.floor(Math.random() * 1000000000000)}-${Math.floor(Math.random() * 1000000000000)}-${Math.random().toString(36).substr(2, 24)}`,
      scope: "chat:write,files:read"
    }
  };
  
  return templates[provider] || {
    access_token: Math.random().toString(36).substr(2, 32),
    type: "bearer"
  };
}