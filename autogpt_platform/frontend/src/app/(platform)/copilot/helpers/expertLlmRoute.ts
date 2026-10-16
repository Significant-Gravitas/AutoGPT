import type { Expert } from "@/app/api/__generated__/models/expert";
import type { CopilotLlmAuthSelection } from "../store";

/**
 * The connection an expert's new threads start on, as a selection the picker
 * and the session hook understand. Null when the expert follows the account
 * default or when its pin names a connection the owner can no longer chat
 * over — the server falls back in that case, and so does the client.
 */
export function getExpertLlmRoute(
  expert: Pick<
    Expert,
    "llm_auth_provider" | "llm_credential_id" | "llm_route_available"
  > | null,
): CopilotLlmAuthSelection | null {
  if (!expert?.llm_auth_provider) return null;
  if (expert.llm_route_available === false) return null;
  if (expert.llm_auth_provider === "platform") {
    return { authProvider: "platform", credentialId: null };
  }
  if (!expert.llm_credential_id) return null;
  return {
    authProvider: expert.llm_auth_provider,
    credentialId: expert.llm_credential_id,
  };
}
