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

  if (error?.code === "P0001") return true;

  return (
    error.message.includes("P0001") || // PostgreSQL custom error code
    error.message.includes("not allowed to register") || // Trigger message
    error.message.toLowerCase().includes("allowed_users") // Table reference
  );
}

/**
 * Checks if OAuth callback URL parameters indicate a waitlist error
 *
 * This is for the auth-code-error page which receives errors via URL hash params
 * from Supabase OAuth redirects
 *
 * @param errorCode - The error_code parameter from the URL
 * @param errorDescription - The error_description parameter from the URL
 * @returns true if this appears to be a waitlist/allowlist error
 */
export function isWaitlistErrorFromParams(
  errorCode?: string | null,
  errorDescription?: string | null,
): boolean {
  if (!errorDescription) return false;

  if (errorCode === "P0001") return true;

  const description = errorDescription.toLowerCase();
  return (
    description.includes("p0001") || // PostgreSQL error code might be in description
    description.includes("not allowed") ||
    description.includes("waitlist") ||
    description.includes("allowlist") ||
    description.includes("allowed_users")
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
