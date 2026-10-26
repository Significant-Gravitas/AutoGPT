export function WebhookTriggerBanner() {
  return (
    <div
      // TODO(tokens): no exact role for border-blue-200 (#BFDBFE)
      className="mb-4 rounded-lg border border-blue-200 bg-info-surface p-4"
    >
      <div className="flex items-start">
        <div className="flex-shrink-0">
          <svg
            // TODO(tokens): no exact role for text-blue-400 (#60A5FA)
            className="h-5 w-5 text-blue-400"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
        </div>
        <div className="ml-3">
          <h3
            // TODO(tokens): no exact role for text-blue-800 (#1E40AF)
            className="text-sm font-medium text-blue-800"
          >
            Webhook Trigger
          </h3>
          <div
            // TODO(tokens): no exact role for text-blue-700 (#1D4ED8)
            className="mt-2 text-sm text-blue-700"
          >
            <p>
              This will create a webhook endpoint that automatically runs your
              workflow when triggered by external events.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
