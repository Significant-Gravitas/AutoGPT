// Pre-authenticated tokens for load testing (EXAMPLE FILE)
// Run: node generate-tokens.js to create the actual configs/pre-authenticated-tokens.js file
// 
// This example shows the expected structure but contains no real tokens

export const PRE_AUTHENTICATED_TOKENS = [
  // {
  //   "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  //   "user": "loadtest1@example.com",
  //   "generated": "2025-09-24T02:33:27.054Z",
  //   "round": 1
  // }
];

export function getPreAuthenticatedToken(vuId = 1) {
  if (PRE_AUTHENTICATED_TOKENS.length === 0) {
    throw new Error('No pre-authenticated tokens available. Run: node generate-tokens.js');
  }
  
  const tokenIndex = (vuId - 1) % PRE_AUTHENTICATED_TOKENS.length;
  const tokenData = PRE_AUTHENTICATED_TOKENS[tokenIndex];
  
  return {
    access_token: tokenData.token,
    user: { email: tokenData.user },
    generated: tokenData.generated
  };
}

export function getPreAuthenticatedHeaders(vuId = 1) {
  const authData = getPreAuthenticatedToken(vuId);
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${authData.access_token}`,
  };
}

export const TOKEN_STATS = {
  total: PRE_AUTHENTICATED_TOKENS.length,
  users: [...new Set(PRE_AUTHENTICATED_TOKENS.map(t => t.user))].length,
  generated: PRE_AUTHENTICATED_TOKENS[0]?.generated || 'unknown'
};

console.log(`🔐 Loaded ${TOKEN_STATS.total} pre-authenticated tokens from ${TOKEN_STATS.users} users`);