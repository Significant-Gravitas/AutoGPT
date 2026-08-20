import { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { useContext } from "react";
import { getCredentialProviderFromSchema } from "./helpers";

type Availability = "loading" | "available" | "unavailable";

/**
 * Whether this field's provider is usable by the current user.
 *
 * The providers map is null until the provider-name and credential fetches
 * settle, and the backend omits providers the user isn't entitled to (e.g.
 * codex below the MAX tier). Those are different states: null means "unknown
 * yet", a loaded map missing the key means "this user cannot connect it".
 * Collapsing them would flash an unavailable state at entitled users on every
 * page load.
 */
export function useCredentialAvailability(
  schema: BlockIOCredentialsSubSchema,
  formData: Record<string, unknown>,
  selectedProvider?: string,
): Availability {
  const providers = useContext(CredentialsProvidersContext);
  const provider = getCredentialProviderFromSchema(
    formData,
    schema,
    selectedProvider,
  );

  if (providers === null) return "loading";
  // A multi-provider field with no discriminator value yet resolves to null;
  // nothing is decided, so don't claim unavailability.
  if (!provider) return "loading";

  return providers[provider] ? "available" : "unavailable";
}
