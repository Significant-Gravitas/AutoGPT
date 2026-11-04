import { ApiError } from "@/lib/autogpt-server-api/helpers";

// What a team organization's members read when a refusal gives no reason of
// its own: the organization's plan is set up through AutoGPT, not bought here.
export const TEAM_ORG_PLAN_REQUIRED =
  "This organization needs a team plan. An organization owner can contact AutoGPT to set one up.";

// The backend's reason for refusing with a 402, when it gave one: an API
// error's `detail`, or that of the JSON body a refused chat send carries as
// its message.
export function paywallDetail(error: unknown) {
  if (error instanceof ApiError) return stringDetail(error.response);
  if (error instanceof Error) return stringDetail(parseBody(error.message));
  return undefined;
}

function stringDetail(body: unknown) {
  if (typeof body !== "object" || body === null || !("detail" in body)) {
    return undefined;
  }
  const { detail } = body;
  return typeof detail === "string" && detail.trim() ? detail : undefined;
}

function parseBody(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}
