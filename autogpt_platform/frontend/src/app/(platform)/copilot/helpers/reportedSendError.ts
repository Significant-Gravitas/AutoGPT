// A send failure whose cause the user has already been shown (a toast or an
// inline notice). The composer still restores the draft, but adds no second
// toast that would only repeat the error.
export class ReportedSendError extends Error {
  readonly name = "ReportedSendError";
}

export const SELF_HOSTED_MODELS_DOCS_URL =
  "https://github.com/Significant-Gravitas/AutoGPT/blob/master/docs/platform/single-container.md#models-and-memory";
