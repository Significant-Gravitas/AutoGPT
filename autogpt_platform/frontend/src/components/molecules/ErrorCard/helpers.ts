import { ErrorCardProps } from "./ErrorCard";

export function getErrorMessage(
  error: ErrorCardProps["responseError"],
): string {
  if (!error) return "Unknown error occurred";

  if (typeof error.detail === "string") return error.detail;
  if (Array.isArray(error.detail) && error.detail.length > 0) {
    return error.detail[0].msg;
  }
  if (error.message) return error.message;

  return "Unknown error occurred";
}

export function getHttpErrorMessage(
  httpError: ErrorCardProps["httpError"],
): string {
  if (!httpError) return "";

  const status = httpError.status || 0;

  if (status >= 500) {
    return "Our servers are having a bit of a moment 🤖 Please try again in a few minutes.";
  }

  if (status === 404) {
    return "We couldn't find what you're looking for. It might have wandered off somewhere! 🔍";
  }

  if (status === 403) {
    return "You don't have permission to access this. Maybe you need to sign in again? 🔐";
  }

  if (status === 429) {
    return "Whoa there, speed racer! You're making requests too quickly. Take a breather and try again. ⏱️";
  }

  if (status >= 400) {
    return "Something's not quite right with your request. Double-check and try again! ✨";
  }

  return "Something unexpected happened on our end. We're on it! 🛠️";
}

export function shouldShowError(
  isSuccess: boolean,
  responseError?: ErrorCardProps["responseError"],
  httpError?: ErrorCardProps["httpError"],
): boolean {
  return !isSuccess || !!responseError || !!httpError;
}

export function isHttpError(httpError?: ErrorCardProps["httpError"]): boolean {
  return !!httpError;
}

export function handleReportError(
  responseError?: ErrorCardProps["responseError"],
  httpError?: ErrorCardProps["httpError"],
  context?: string,
): void {
  try {
    // Import Sentry dynamically to avoid SSR issues
    import("@sentry/nextjs").then((Sentry) => {
      // Create a comprehensive error object for Sentry
      const errorData = {
        responseError,
        httpError,
        context,
        timestamp: new Date().toISOString(),
        userAgent:
          typeof window !== "undefined" ? window.navigator.userAgent : "server",
        url: typeof window !== "undefined" ? window.location.href : "unknown",
      };

      // Create an error object that Sentry can capture
      const errorMessage = httpError
        ? `HTTP ${httpError.status} - ${httpError.statusText || "Error"}`
        : responseError
          ? `Response Error - ${getErrorMessage(responseError)}`
          : "Unknown Error";

      const error = new Error(
        `ErrorCard: ${context ? `${context} ` : ""}${errorMessage}`,
      );

      // Add extra context to Sentry
      Sentry.withScope((scope) => {
        scope.setTag("component", "ErrorCard");
        scope.setTag("errorType", httpError ? "http" : "response");
        scope.setContext("errorDetails", errorData);

        if (context) {
          scope.setTag("context", context);
        }

        if (httpError?.status) {
          scope.setTag("httpStatus", httpError.status.toString());
        }

        // Capture the exception
        Sentry.captureException(error);
      });

      // Show success toast notification
      import("sonner").then(({ toast }) => {
        toast.success("Error reported successfully", {
          description:
            "Thank you for helping us improve! Our team has been notified.",
          duration: 4000,
        });
      });
    });
  } catch (error) {
    console.error("Failed to report error to Sentry:", error);
    // Fallback toast notification
    import("sonner").then(({ toast }) => {
      toast.error("Failed to report error", {
        description: "Please try again or contact support directly.",
        duration: 4000,
      });
    });
  }
}
