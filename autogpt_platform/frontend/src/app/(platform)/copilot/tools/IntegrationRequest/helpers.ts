import { asObject } from "../../components/ToolChain/resultHelpers";

export interface IntegrationRequest {
  service: string;
  recorded: boolean;
}

/** The `request_integration` response, whether it arrived as the tool's own
 *  part or through `run_capability`; null until it has landed. */
export function parseIntegrationRequest(
  output: unknown,
): IntegrationRequest | null {
  const data = asObject(output);
  if (!data || data.type !== "integration_request") return null;
  if (typeof data.service !== "string" || !data.service.trim()) return null;
  return { service: data.service.trim(), recorded: data.recorded === true };
}
