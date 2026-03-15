/**
 * Checks if an error is related to the waitlist/allowlist
 *
 * Can be used with either:
 * - Error objects from Supabase auth operations: `isWaitlistError(error?.code, error?.message)`
 * - URL parameters from OAuth callbacks: `isWaitlistError(errorCode, errorDescription)`
 *
 * The PostgreSQL trigger raises P0001 with message format:
 * "The email address "email" is not allowed to register. Please contact support for assistance."
 *
 * @param code - Error code (e.g., "P0001", "unexpected_failure") or null
 * @param message - Error message/description or null
 * @returns true if this appears to be a waitlist/allowlist error
 */
export function isWaitlistError(
  code?: string | null,
  message?: string | null,
): boolean {
  // Check for explicit PostgreSQL trigger error code
  if (code === "P0001") return true;

  if (!message) return false;

  const lowerMessage = message.toLowerCase();

  // Check for the generic database error that occurs during waitlist check
  // This happens when Supabase wraps the PostgreSQL trigger error
  if (
    code === "unexpected_failure" &&
    message === "Database error saving new user"
  ) {
    return true;
  }

  // Check for various waitlist-related patterns in the message
  return (
    lowerMessage.includes("p0001") || // PostgreSQL error code in message
    lowerMessage.includes("not allowed") || // Common waitlist message
    lowerMessage.includes("waitlist") || // Explicit waitlist mention
    lowerMessage.includes("allowlist") || // Explicit allowlist mention
    lowerMessage.includes("allowed_users") || // Database table reference
    lowerMessage.includes("not allowed to register") // Full trigger message
  );
}

/**
 * Logs a waitlist error for debugging purposes
 * Does not expose user email in logs for privacy
 *
 * @param context - Where the error occurred (e.g., "Signup", "OAuth Provider")
 * @param errorMessage - The full error message
 */
export function logWaitlistError(context: string, errorMessage: string): void {
  // Only log the error code and general message, not the email
  const sanitizedMessage = errorMessage.replace(
    /"[^"]+@[^"]+"/g, // Matches email addresses in quotes
    '"[email]"',
  );
  console.log(`[${context}] Waitlist check failed:`, sanitizedMessage);
}
