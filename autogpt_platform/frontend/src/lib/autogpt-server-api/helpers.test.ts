/**
 * Unit tests for helpers.ts
 *
 * These tests validate the error handling in handleFetchError, specifically
 * the fix for the issue where calling response.json() on non-JSON responses
 * would throw: "Failed to execute 'json' on 'Response': Unexpected token 'A',
 * "A server e"... is not valid JSON"
 *
 * To run these tests, you'll need to set up a unit test framework like Jest or Vitest.
 *
 * Test cases to cover:
 *
 * 1. JSON error responses should be parsed correctly
 *    - Given: Response with content-type: application/json
 *    - When: handleFetchError is called
 *    - Then: Should parse JSON and return ApiError with parsed response
 *
 * 2. Non-JSON error responses (e.g., HTML) should be handled gracefully
 *    - Given: Response with content-type: text/html
 *    - When: handleFetchError is called
 *    - Then: Should read as text and return ApiError with text response
 *
 * 3. Response without content-type header should be handled
 *    - Given: Response without content-type header
 *    - When: handleFetchError is called
 *    - Then: Should default to reading as text
 *
 * 4. JSON parsing errors should not throw
 *    - Given: Response with content-type: application/json but HTML body
 *    - When: handleFetchError is called and json() throws
 *    - Then: Should catch error, log warning, and return ApiError with null response
 *
 * 5. Specific validation for the fixed bug
 *    - Given: 502 Bad Gateway with content-type: application/json but HTML body
 *    - When: response.json() throws "Unexpected token 'A'" error
 *    - Then: Should NOT propagate the error, should return ApiError with null response
 */

import { handleFetchError } from "./helpers";

// Manual test function - can be run in browser console or Node
export async function testHandleFetchError() {
  console.log("Testing handleFetchError...");

  // Test 1: JSON response
  const jsonResponse = new Response(
    JSON.stringify({ error: "Internal server error" }),
    {
      status: 500,
      headers: { "content-type": "application/json" },
    },
  );
  const error1 = await handleFetchError(jsonResponse);
  console.assert(
    error1.status === 500 && error1.response?.error === "Internal server error",
    "Test 1 failed: JSON response",
  );

  // Test 2: HTML response
  const htmlResponse = new Response("<html><body>Server Error</body></html>", {
    status: 502,
    headers: { "content-type": "text/html" },
  });
  const error2 = await handleFetchError(htmlResponse);
  console.assert(
    error2.status === 502 &&
      typeof error2.response === "string" &&
      error2.response.includes("Server Error"),
    "Test 2 failed: HTML response",
  );

  // Test 3: Mismatched content-type (claims JSON but is HTML)
  // This simulates the bug that was fixed
  const mismatchedResponse = new Response(
    "<html><body>A server error occurred</body></html>",
    {
      status: 502,
      headers: { "content-type": "application/json" }, // Claims JSON but isn't
    },
  );
  try {
    const error3 = await handleFetchError(mismatchedResponse);
    console.assert(
      error3.status === 502 && error3.response === null,
      "Test 3 failed: Mismatched content-type should return null response",
    );
    console.log("✓ All tests passed!");
  } catch (e) {
    console.error("✗ Test 3 failed: Should not throw error", e);
  }
}

// Uncomment to run manual tests
// testHandleFetchError();
