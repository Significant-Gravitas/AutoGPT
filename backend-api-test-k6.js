import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';
import { randomItem } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');

// Vercel Protection Bypass Secret
const VERCEL_BYPASS_SECRET = 'pasjfobnqw0rujeqnfoasbnr0q23jqsf';

// Test configuration
export const options = {

    stages: [
        { duration: '10m', target: 1000 },  // Ramp up to 100 users over 5 minutes
    ],
    thresholds: {
        http_req_duration: [
            'p(90)<1000',  // 90% of requests should be below 1s
            'p(95)<2000',  // 95% of requests should be below 2s
        ],
        http_req_failed: ['rate<0.05'],      // Error rate should be less than 5%
        errors: ['rate<0.05'],               // Custom error rate should be less than 5%
    },
};

// Base URL - using the proxy endpoint
const BASE_URL = 'https://autogpt-env-staging-significant-gravitas.vercel.app/api/proxy/api/store';

// Define different API endpoints with their weights
const endpoints = {
    // List agents endpoints
    allAgents: {
        url: `${BASE_URL}/agents`,
        weight: 20,
        name: 'All Agents',
    },
    featuredAgents: {
        url: `${BASE_URL}/agents?featured=true`,
        weight: 15,
        name: 'Featured Agents',
    },
    paginatedAgents: [
        {
            url: `${BASE_URL}/agents?page=1&page_size=20`,
            weight: 10,
            name: 'Agents Page 1',
        },
        {
            url: `${BASE_URL}/agents?page=2&page_size=20`,
            weight: 5,
            name: 'Agents Page 2',
        },
    ],
    // Creator-specific agents
    creatorAgents: [
        {
            url: `${BASE_URL}/agents?creator=hey`,
            weight: 10,
            name: 'Hey Creator Agents',
        },
        {
            url: `${BASE_URL}/agents?creator=exaai`,
            weight: 10,
            name: 'ExaAI Creator Agents',
        },
        {
            url: `${BASE_URL}/agents?creator=swiftyos`,
            weight: 5,
            name: 'SwiftyOS Creator Agents',
        },
    ],
    // Individual agent details - requires username/agent_name format
    specificAgents: [
        {
            url: `${BASE_URL}/agents/hey/release-note-generator`,
            weight: 10,
            name: 'Release Note Generator Agent',
        },
        {
            url: `${BASE_URL}/agents/swiftyos/agent-test`,
            weight: 5,
            name: 'Agent Test',
        },
    ],
    // Creator profiles
    creatorProfiles: [
        {
            url: `${BASE_URL}/creator/hey`,  // Note: singular 'creator' not 'creators'
            weight: 10,
            name: 'Hey Creator Profile',
        },
        {
            url: `${BASE_URL}/creator/exaai`,
            weight: 10,
            name: 'ExaAI Creator Profile',
        },
    ],
    // List all creators
    creatorsList: {
        url: `${BASE_URL}/creators`,
        weight: 10,
        name: 'All Creators',
    },
};

// Search terms for dynamic searches
const searchTerms = [
    'Marketing',
    'Find',
    'AI',
    'Assistant',
    'Generator',
    'Analytics',
    'Automation',
    'Data',
    'Report',
    'Tool',
    'Calculator',
    'Business',
    'Email',
    'Social',
    'Content',
];

// Sort options for the API
const sortOptions = ['runs', 'rating'];

// Categories available
const categories = [
    'business',
    'productivity',
    'marketing',
    'development',
    'data',
    'creative',
    'other',
];

// Helper function to get a weighted random endpoint
function getRandomEndpoint() {
    const rand = Math.random() * 100;

    if (rand < 20) {
        // Get all agents
        return endpoints.allAgents;
    } else if (rand < 35) {
        // Get featured agents
        return endpoints.featuredAgents;
    } else if (rand < 45) {
        // Get paginated agents
        return randomItem(endpoints.paginatedAgents);
    } else if (rand < 60) {
        // Get creator-specific agents
        return randomItem(endpoints.creatorAgents);
    } else if (rand < 70) {
        // Get specific agent details or creator profile
        if (Math.random() < 0.5) {
            return randomItem(endpoints.specificAgents);
        } else {
            return randomItem(endpoints.creatorProfiles);
        }
    } else if (rand < 75) {
        // Get all creators list
        return endpoints.creatorsList;
    } else {
        // Perform a dynamic search
        return generateSearchEndpoint();
    }
}

// Generate dynamic search endpoint
function generateSearchEndpoint() {
    const term = randomItem(searchTerms);
    let url = `${BASE_URL}/agents?search_query=${encodeURIComponent(term)}`;

    // 30% chance to add sorting
    if (Math.random() < 0.3) {
        const sort = randomItem(sortOptions);
        url += `&sorted_by=${sort}`;
    }

    // 20% chance to add category filter
    if (Math.random() < 0.2) {
        const category = randomItem(categories);
        url += `&category=${category}`;
    }

    // 40% chance to add pagination
    if (Math.random() < 0.4) {
        const page = Math.floor(Math.random() * 3) + 1;
        url += `&page=${page}&page_size=20`;
    }

    return {
        url,
        name: `Search: ${term}`,
    };
}

// User session simulation for API testing
function testBackendAPIs() {
    const params = {
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'k6-load-test/1.0',
            'x-vercel-protection-bypass': VERCEL_BYPASS_SECRET,  // Added bypass header
        },
        timeout: '30s',
        tags: { name: 'backend-api-test' },
    };

    // Start with the main agents list
    let res = http.get(endpoints.allAgents.url, params);

    check(res, {
        'agents list status is 200': (r) => r.status === 200,
        'agents list has valid JSON': (r) => {
            try {
                const body = JSON.parse(r.body);
                return body.agents !== undefined && body.pagination !== undefined;
            } catch {
                return false;
            }
        },
        'agents list loaded quickly': (r) => r.timings.duration < 1000,
    }) || errorRate.add(1);

    // Think time - simulate processing delay
    sleep(randomBetween(0.5, 2));

    // Make 5-10 additional API calls
    const callsToMake = Math.floor(Math.random() * 6) + 5;

    for (let i = 0; i < callsToMake; i++) {
        const endpoint = getRandomEndpoint();

        res = http.get(endpoint.url, params);

        // Determine endpoint type
        const isAgentsList = endpoint.url.includes('/agents') && !endpoint.url.match(/\/agents\/[^\/]+\/[^\/]+$/);
        const isAgentDetail = endpoint.url.match(/\/agents\/[^\/]+\/[^\/]+$/);
        const isCreatorsList = endpoint.url.endsWith('/creators');
        const isCreatorDetail = endpoint.url.match(/\/creator\/[^\/]+$/);

        // Different checks based on endpoint type
        if (isAgentsList) {
            check(res, {
                [`${endpoint.name} status is 200`]: (r) => r.status === 200,
                'response has agents array': (r) => {
                    try {
                        const body = JSON.parse(r.body);
                        return Array.isArray(body.agents);
                    } catch {
                        return false;
                    }
                },
                'API response time within SLA': (r) => r.timings.duration < 1000,
            }) || errorRate.add(1);
        } else if (isAgentDetail) {
            check(res, {
                [`${endpoint.name} status is 200`]: (r) => r.status === 200,
                'API response time within SLA': (r) => r.timings.duration < 1000,
            }) || errorRate.add(1);
        } else if (isCreatorsList) {
            check(res, {
                [`${endpoint.name} status is 200`]: (r) => r.status === 200,
                'response has creators array': (r) => {
                    try {
                        const body = JSON.parse(r.body);
                        return Array.isArray(body.creators);
                    } catch {
                        return false;
                    }
                },
                'API response time within SLA': (r) => r.timings.duration < 1000,
            }) || errorRate.add(1);
        } else if (isCreatorDetail) {
            check(res, {
                [`${endpoint.name} status is 200`]: (r) => r.status === 200,
                'creator has profile data': (r) => {
                    try {
                        const body = JSON.parse(r.body);
                        // CreatorDetails should have username and other fields
                        return body && (body.username || body.name);
                    } catch {
                        return false;
                    }
                },
                'API response time within SLA': (r) => r.timings.duration < 1000,
            }) || errorRate.add(1);
        }

        // Log response details for debugging
        if (res.status !== 200) {
            console.log(`Error on ${endpoint.name}: Status ${res.status}`);
        }

        // Variable think time based on operation
        if (isAgentDetail) {
            // Detailed agent view might trigger more processing
            sleep(randomBetween(1, 3));
        } else if (endpoint.url.includes('search_query')) {
            // Search results are processed quickly
            sleep(randomBetween(0.5, 1.5));
        } else {
            // Default think time
            sleep(randomBetween(0.5, 2));
        }

        // Simulate some users ending their session early
        if (Math.random() < 0.1) {
            break;
        }
    }

}

// Helper function for random sleep times
function randomBetween(min, max) {
    return Math.random() * (max - min) + min;
}

// Main test scenario
export default function () {
    testBackendAPIs();

    // Session think time - time between user sessions
    sleep(randomBetween(2, 5));
}

// Setup function - runs once before the test
export function setup() {
    console.log('Starting backend API load test...');
    console.log(`Target: ${BASE_URL}`);
    console.log('Using Vercel protection bypass for staging environment');
    console.log('Testing endpoints:');
    console.log('- /agents (listings with filtering/search/pagination)');
    console.log('- /agents/{username}/{agent_name} (agent details)');
    console.log('- /creators (list all creators)');
    console.log('- /creator/{username} (creator profile)');
    console.log('- Search queries with various filters');

    // Setup params with bypass header
    const setupParams = {
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'x-vercel-protection-bypass': VERCEL_BYPASS_SECRET,
        },
    };

    // Verify the API is accessible
    const res = http.get(`${BASE_URL}/agents?page_size=1`, setupParams);
    if (res.status !== 200) {
        console.log(`Response status: ${res.status}`);
        console.log(`Response body: ${res.body}`);
        throw new Error(`Backend API is not accessible. Status: ${res.status}`);
    }

    try {
        const body = JSON.parse(res.body);
        if (!body.agents || !body.pagination) {
            throw new Error('Invalid API response structure');
        }
        console.log(`✅ API verified. Total agents available: ${body.pagination.total}`);
    } catch (e) {
        throw new Error(`API response validation failed: ${e.message}`);
    }

    return { startTime: new Date().toISOString() };
}

// Teardown function - runs once after the test
export function teardown(data) {
    console.log(`Test completed. Started at: ${data.startTime}`);
    console.log('Check the k6 cloud or output for detailed metrics.');
}