#!/usr/bin/env node

/**
 * Generate Pre-Authenticated Tokens for Load Testing
 * Creates configs/pre-authenticated-tokens.js with 350+ tokens
 *
 * This replaces the old token generation scripts with a clean, single script
 */

import https from "https";
import fs from "fs";
import path from "path";

// Get Supabase service key from environment (REQUIRED for token generation)
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

if (!SUPABASE_SERVICE_KEY) {
  console.error("❌ SUPABASE_SERVICE_KEY environment variable is required");
  console.error("Get service key from kubectl or environment:");
  console.error('export SUPABASE_SERVICE_KEY="your-service-key"');
  process.exit(1);
}

// Generate test users (loadtest4-50 are known to work)
const TEST_USERS = [];
for (let i = 4; i <= 50; i++) {
  TEST_USERS.push({
    email: `loadtest${i}@example.com`,
    password: "password123",
  });
}

console.log(
  `🔐 Generating pre-authenticated tokens from ${TEST_USERS.length} users...`,
);

async function authenticateUser(user, attempt = 1) {
  return new Promise((resolve) => {
    const postData = JSON.stringify({
      email: user.email,
      password: user.password,
      expires_in: 86400, // 24 hours in seconds (24 * 60 * 60)
    });

    const options = {
      hostname: "adfjtextkuilwuhzdjpf.supabase.co",
      path: "/auth/v1/token?grant_type=password",
      method: "POST",
      headers: {
        Authorization: `Bearer ${SUPABASE_SERVICE_KEY}`,
        apikey: SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
        "Content-Length": postData.length,
      },
    };

    const req = https.request(options, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        try {
          if (res.statusCode === 200) {
            const authData = JSON.parse(data);
            resolve(authData.access_token);
          } else if (res.statusCode === 429) {
            // Rate limited - wait and retry
            console.log(
              `⏳ Rate limited for ${user.email}, waiting 5s (attempt ${attempt}/3)...`,
            );
            setTimeout(() => {
              if (attempt < 3) {
                authenticateUser(user, attempt + 1).then(resolve);
              } else {
                console.log(`❌ Max retries exceeded for ${user.email}`);
                resolve(null);
              }
            }, 5000);
          } else {
            console.log(`❌ Auth failed for ${user.email}: ${res.statusCode}`);
            resolve(null);
          }
        } catch (e) {
          console.log(`❌ Parse error for ${user.email}:`, e.message);
          resolve(null);
        }
      });
    });

    req.on("error", (err) => {
      console.log(`❌ Request error for ${user.email}:`, err.message);
      resolve(null);
    });

    req.write(postData);
    req.end();
  });
}

async function generateTokens() {
  console.log("🚀 Starting token generation...");
  console.log("Rate limit aware - this will take ~10-15 minutes");
  console.log("===========================================\n");

  const tokens = [];
  const startTime = Date.now();

  // Generate tokens - configurable via --count argument or default to 150
  const targetTokens =
    parseInt(
      process.argv.find((arg) => arg.startsWith("--count="))?.split("=")[1],
    ) ||
    parseInt(process.env.TOKEN_COUNT) ||
    150;
  const tokensPerUser = Math.ceil(targetTokens / TEST_USERS.length);
  console.log(
    `📊 Generating ${tokensPerUser} tokens per user (${TEST_USERS.length} users) - Target: ${targetTokens}\n`,
  );

  for (let round = 1; round <= tokensPerUser; round++) {
    console.log(`🔄 Round ${round}/${tokensPerUser}:`);

    for (
      let i = 0;
      i < TEST_USERS.length && tokens.length < targetTokens;
      i++
    ) {
      const user = TEST_USERS[i];

      process.stdout.write(`   ${user.email.padEnd(25)} ... `);

      const token = await authenticateUser(user);

      if (token) {
        tokens.push({
          token,
          user: user.email,
          generated: new Date().toISOString(),
          round: round,
        });
        console.log(`✅ (${tokens.length}/${targetTokens})`);
      } else {
        console.log(`❌`);
      }

      // Respect rate limits - wait 500ms between requests
      if (tokens.length < targetTokens) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }

    if (tokens.length >= targetTokens) break;

    // Wait longer between rounds
    if (round < tokensPerUser) {
      console.log(`   ⏸️  Waiting 3s before next round...\n`);
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }

  const duration = Math.round((Date.now() - startTime) / 1000);
  console.log(`\n✅ Generated ${tokens.length} tokens in ${duration}s`);

  // Create configs directory if it doesn't exist
  const configsDir = path.join(process.cwd(), "configs");
  if (!fs.existsSync(configsDir)) {
    fs.mkdirSync(configsDir, { recursive: true });
  }

  // Write tokens to secure file
  const jsContent = `// Pre-authenticated tokens for load testing
// Generated: ${new Date().toISOString()}
// Total tokens: ${tokens.length}
// Generation time: ${duration} seconds
// 
// ⚠️  SECURITY: This file contains real authentication tokens
// ⚠️  DO NOT COMMIT TO GIT - File is gitignored

export const PRE_AUTHENTICATED_TOKENS = ${JSON.stringify(tokens, null, 2)};

export function getPreAuthenticatedToken(vuId = 1) {
  if (PRE_AUTHENTICATED_TOKENS.length === 0) {
    throw new Error('No pre-authenticated tokens available');
  }
  
  const tokenIndex = (vuId - 1) % PRE_AUTHENTICATED_TOKENS.length;
  const tokenData = PRE_AUTHENTICATED_TOKENS[tokenIndex];
  
  return {
    access_token: tokenData.token,
    user: { email: tokenData.user },
    generated: tokenData.generated
  };
}

// Generate single session ID for this test run
const LOAD_TEST_SESSION_ID = '${new Date().toISOString().slice(0, 16).replace(/:/g, "-")}-' + Math.random().toString(36).substr(2, 8);

export function getPreAuthenticatedHeaders(vuId = 1) {
  const authData = getPreAuthenticatedToken(vuId);
  
  return {
    'Content-Type': 'application/json',
    'Authorization': \`Bearer \${authData.access_token}\`,
    'X-Load-Test-Session': LOAD_TEST_SESSION_ID,
    'X-Load-Test-VU': vuId.toString(),
    'X-Load-Test-User': authData.user.email,
  };
}

export const TOKEN_STATS = {
  total: PRE_AUTHENTICATED_TOKENS.length,
  users: [...new Set(PRE_AUTHENTICATED_TOKENS.map(t => t.user))].length,
  generated: PRE_AUTHENTICATED_TOKENS[0]?.generated || 'unknown'
};

console.log(\`🔐 Loaded \${TOKEN_STATS.total} pre-authenticated tokens from \${TOKEN_STATS.users} users\`);
`;

  const tokenFile = path.join(configsDir, "pre-authenticated-tokens.js");
  fs.writeFileSync(tokenFile, jsContent);

  console.log(`💾 Saved to configs/pre-authenticated-tokens.js`);
  console.log(`🚀 Ready for ${tokens.length} concurrent VU load testing!`);
  console.log(
    `\n🔒 Security Note: Token file is gitignored and will not be committed`,
  );

  return tokens.length;
}

// Run if called directly
if (process.argv[1] === new URL(import.meta.url).pathname) {
  generateTokens().catch(console.error);
}

export { generateTokens };
