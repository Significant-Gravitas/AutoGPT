// Pre-authenticated tokens for load testing (EXAMPLE FILE)
// Copy this to pre-authenticated-tokens.js and run generate-tokens.js to populate
//
// ⚠️  SECURITY: The real file contains authentication tokens
// ⚠️  DO NOT COMMIT TO GIT - Real file is gitignored

export const PRE_AUTHENTICATED_TOKENS = [
  // Will be populated by generate-tokens.js with 350+ real tokens
  // Example structure:
  // {
  //   token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  //   user: "loadtest4@example.com",
  //   generated: "2025-01-24T10:08:04.123Z",
  //   round: 1
  // }
];

export function getPreAuthenticatedToken(vuId = 1) {
  if (PRE_AUTHENTICATED_TOKENS.length === 0) {
    throw new Error(
      "No pre-authenticated tokens available. Run: node generate-tokens.js",
    );
  }

  const tokenIndex = (vuId - 1) % PRE_AUTHENTICATED_TOKENS.length;
  const tokenData = PRE_AUTHENTICATED_TOKENS[tokenIndex];

  return {
    access_token: tokenData.token,
    user: { email: tokenData.user },
    generated: tokenData.generated,
  };
}

export function getPreAuthenticatedHeaders(vuId = 1) {
  const authData = getPreAuthenticatedToken(vuId);
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${authData.access_token}`,
  };
}

export const TOKEN_STATS = {
  total: PRE_AUTHENTICATED_TOKENS.length,
  users: [...new Set(PRE_AUTHENTICATED_TOKENS.map((t) => t.user))].length,
  generated: PRE_AUTHENTICATED_TOKENS[0]?.generated || "unknown",
};

console.log(
  `🔐 Loaded ${TOKEN_STATS.total} pre-authenticated tokens from ${TOKEN_STATS.users} users`,
);
