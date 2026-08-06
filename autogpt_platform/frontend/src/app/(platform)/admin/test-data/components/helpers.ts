const FALLBACK_ERROR_DETAIL = "Failed to generate test data. Please try again.";

export function getErrorDetail(error: unknown) {
  if (error instanceof Error && error.message) return error.message;
  return FALLBACK_ERROR_DETAIL;
}
