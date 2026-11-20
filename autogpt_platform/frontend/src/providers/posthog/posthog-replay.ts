import type { CapturedNetworkRequest } from "posthog-js";

// Session replays record each request the page makes by its address alone:
// the query string, the fragment and the request and response bodies never
// reach PostHog.
export function maskCapturedNetworkRequest(
  request: CapturedNetworkRequest,
): CapturedNetworkRequest {
  return {
    ...request,
    name: addressOf(request.name),
    requestBody: undefined,
    responseBody: undefined,
  };
}

function addressOf(url: string) {
  const end = url.search(/[?#]/);
  return end === -1 ? url : url.slice(0, end);
}
