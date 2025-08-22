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
    return "An internal server error has occurred. Please try again in a few minutes.";
  }

  if (status === 404) {
    return "The requested resource could not be found. Please verify the URL and try again.";
  }

  if (status === 403) {
    return "Access to this resource is forbidden. Please check your permissions or sign in again.";
  }

  if (status === 429) {
    return "Too many requests have been made. Please wait a moment before trying again.";
  }

  if (status >= 400) {
    return "The request could not be processed. Please review your input and try again.";
  }

  return "An unexpected error has occurred. Our team has been notified and is working to resolve the issue.";
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

        Sentry.captureException(error);
      });

      // Show success toast notification after pressing the report error button
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
