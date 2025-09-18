// Simple script to find block IDs
import http from 'k6/http';
import { getEnvironmentConfig } from './configs/environment.js';
import { authenticateUser, getAuthHeaders, getRandomTestUser } from './utils/auth.js';

const config = getEnvironmentConfig();

export const options = {
  stages: [{ duration: '5s', target: 1 }],
};

export default function () {
  const testUser = getRandomTestUser();
  const userAuth = authenticateUser(testUser);
  const headers = getAuthHeaders(userAuth.access_token);
  
  const response = http.get(`${config.API_BASE_URL}/api/blocks`, { headers });
  
  if (response.status === 200) {
    const blocks = JSON.parse(response.body);
    
    // Find time-related blocks
    const timeBlocks = blocks.filter(block => 
      block.name.toLowerCase().includes('time') ||
      block.name.toLowerCase().includes('date')
    );
    
    console.log(`\n🔍 Found ${timeBlocks.length} time-related blocks:`);
    timeBlocks.forEach(block => {
      console.log(`  ID: ${block.id}`);
      console.log(`  Name: ${block.name}`);
      console.log(`  ---`);
    });
    
    // Get a simple block for testing
    const simpleBlocks = blocks.filter(block => 
      block.name.includes('Text') ||
      block.name.includes('Input') ||
      block.name.includes('Output') ||
      block.name.includes('Echo') ||
      block.name.includes('Print')
    );
    
    console.log(`\n📝 Found ${simpleBlocks.length} simple text blocks:`);
    simpleBlocks.slice(0, 3).forEach(block => {
      console.log(`  ID: ${block.id}`);
      console.log(`  Name: ${block.name}`);
      console.log(`  ---`);
    });
    
  } else {
    console.log(`❌ Failed to get blocks: ${response.status}`);
  }
}