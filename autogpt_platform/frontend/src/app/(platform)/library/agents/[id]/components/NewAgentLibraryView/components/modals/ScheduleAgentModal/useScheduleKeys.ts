import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { useContext, useState } from "react";
import { getReplacedKeys } from "./helpers";

/**
 * The keys a schedule runs on: its own, which can differ from the ones the
 * run form shows, or the run form's once the user picks them.
 */
export function useScheduleKeys(
  scheduleCredentials: Record<string, any>,
  formCredentials?: Record<string, any>,
) {
  const allProviders = useContext(CredentialsProvidersContext);
  const [usesFormKeys, setUsesFormKeys] = useState(false);

  return {
    credentials:
      usesFormKeys && formCredentials ? formCredentials : scheduleCredentials,
    replacedKeys: getReplacedKeys({
      scheduleCredentials,
      formCredentials,
      allProviders,
    }),
    usesFormKeys,
    handleUseFormKeys: () => setUsesFormKeys(true),
    resetKeys: () => setUsesFormKeys(false),
  };
}
