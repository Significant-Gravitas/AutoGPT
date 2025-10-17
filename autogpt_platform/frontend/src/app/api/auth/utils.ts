/**
 * Checks if a Supabase auth error is related to the waitlist/allowlist
 *
 * The PostgreSQL trigger raises P0001 with message format:
 * "The email address "email" is not allowed to register. Please contact support for assistance."
 *
 * @param error - The error object from Supabase auth operations
 * @returns true if this is a waitlist/allowlist error
 */
export function isWaitlistError(error: any): boolean {
  if (!error?.message) return false;

  return (
    error.message.includes("P0001") || // PostgreSQL custom error code
    error.message.includes("not allowed to register") || // Trigger message
    error.message.toLowerCase().includes("allowed_users") // Table reference
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
