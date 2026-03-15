export function getErrorDetails(errorType: string | null) {
  switch (errorType) {
    case "user-creation-failed":
      return {
        responseError: {
          message:
            "Failed to create your user account in our system. This could be due to a temporary server issue or a problem with your account setup.",
        },
        context: "user account creation",
      };
    case "auth-token-invalid":
      return {
        responseError: {
          message:
            "Your authentication token is missing or invalid. Please try signing in again.",
        },
        context: "authentication token",
      };
    case "server-error":
      return {
        responseError: {
          message:
            "Our servers are experiencing issues. Please try again in a few minutes, or contact support if the problem persists.",
        },
        context: "server error",
      };
    case "rate-limited":
      return {
        responseError: {
          message:
            "Too many requests have been made. Please wait a moment before trying again.",
        },
        context: "rate limiting",
      };
    case "network-error":
      return {
        responseError: {
          message:
            "Unable to connect to our servers. Please check your internet connection and try again.",
        },
        context: "network connectivity",
      };
    case "auth-failed":
      return {
        responseError: {
          message: "Authentication failed. Please try signing in again.",
        },
        context: "authentication",
      };
    case "session-expired":
      return {
        responseError: {
          message:
            "Your session has expired. Please sign in again to continue.",
        },
        context: "session",
      };
    default:
      return {
        responseError: {
          message:
            "An unexpected error occurred. Please try again or contact support if the problem persists.",
        },
        context: "application",
      };
  }
}
