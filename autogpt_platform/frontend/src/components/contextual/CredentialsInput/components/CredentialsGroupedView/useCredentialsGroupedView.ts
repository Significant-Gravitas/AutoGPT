import {
  CredentialsMetaInput,
  CredentialsType,
} from "@/lib/autogpt-server-api/types";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  areSystemCredentialProvidersLoading,
  CredentialField,
  findSavedCredentialByProviderAndType,
  hasMissingRequiredSystemCredentials,
  splitCredentialFieldsBySystem,
} from "./helpers";
import {
  findNodeCredential,
  isOwnCredential,
  toCredentialsInput,
} from "./node-credentials";

interface Args {
  credentialFields: CredentialField[];
  requiredCredentials: Set<string>;
  inputCredentials: Record<string, CredentialsMetaInput | undefined>;
  onCredentialChange: (key: string, value?: CredentialsMetaInput) => void;
  nodeCredentials: CredentialsMetaInput[];
}

export function useCredentialsGroupedView({
  credentialFields,
  requiredCredentials,
  inputCredentials,
  onCredentialChange,
  nodeCredentials,
}: Args) {
  const allProviders = useContext(CredentialsProvidersContext);

  const { userCredentialFields, systemCredentialFields } = useMemo(
    () => splitCredentialFieldsBySystem(credentialFields, allProviders),
    [credentialFields, allProviders],
  );

  const hasSystemCredentials = systemCredentialFields.length > 0;
  const hasAttemptedAutoSelect = useRef(false);
  // A field the user changes stays where it is: one shown above the system
  // credentials keeps its place if they switch it to the platform key, and
  // one changed inside the open section waits there until it closes.
  const [pinnedAbove, setPinnedAbove] = useState<string[]>([]);
  const [pinnedInside, setPinnedInside] = useState<string[]>([]);

  // A field set to one of the user's own keys is shown above the collapsed
  // system credentials, so they can see which key the run will use.
  function isShownAbove(key: string) {
    if (pinnedAbove.includes(key)) return true;
    if (pinnedInside.includes(key)) return false;
    return isOwnCredential(inputCredentials?.[key], allProviders);
  }

  const promotedFields = systemCredentialFields.filter(([key]) =>
    isShownAbove(key),
  );
  const collapsedFields = systemCredentialFields.filter(
    ([key]) => !isShownAbove(key),
  );

  const isLoadingProviders = useMemo(
    () =>
      areSystemCredentialProvidersLoading(systemCredentialFields, allProviders),
    [systemCredentialFields, allProviders],
  );

  const hasMissingSystemCredentials =
    !isLoadingProviders &&
    hasMissingRequiredSystemCredentials(
      collapsedFields,
      requiredCredentials,
      inputCredentials,
      allProviders,
    );

  function handleFieldChange(key: string, value?: CredentialsMetaInput) {
    if (systemCredentialFields.some(([systemKey]) => systemKey === key)) {
      const pin = isShownAbove(key) ? setPinnedAbove : setPinnedInside;
      pin((keys) => (keys.includes(key) ? keys : [...keys, key]));
    }
    onCredentialChange(key, value);
  }

  function handleSystemSectionChange(openSection: string) {
    if (!openSection) setPinnedInside([]);
  }

  useEffect(() => {
    if (hasAttemptedAutoSelect.current) return;
    if (!hasSystemCredentials && nodeCredentials.length === 0) return;
    if (!allProviders || isLoadingProviders) return;

    for (const [key, schema] of credentialFields) {
      if (inputCredentials?.[key]) continue;

      const nodeCredential = findNodeCredential(
        schema,
        nodeCredentials,
        allProviders,
      );
      if (nodeCredential) {
        onCredentialChange(key, toCredentialsInput(nodeCredential));
        continue;
      }

      const isSystemField = systemCredentialFields.some(
        ([systemKey]) => systemKey === key,
      );
      if (!isSystemField || !requiredCredentials.has(key)) continue;

      const savedCredential = findSavedCredentialByProviderAndType(
        schema.credentials_provider || [],
        schema.credentials_types || [],
        schema.credentials_scopes,
        allProviders,
        schema.discriminator_values,
      );
      if (savedCredential) {
        onCredentialChange(key, {
          id: savedCredential.id,
          provider: savedCredential.provider,
          type: savedCredential.type as CredentialsType,
          title: savedCredential.title,
        });
      }
    }

    hasAttemptedAutoSelect.current = true;
  }, [
    allProviders,
    credentialFields,
    hasSystemCredentials,
    inputCredentials,
    isLoadingProviders,
    nodeCredentials,
    onCredentialChange,
    requiredCredentials,
    systemCredentialFields,
  ]);

  return {
    visibleFields: [...userCredentialFields, ...promotedFields],
    collapsedFields,
    hasMissingSystemCredentials,
    handleFieldChange,
    handleSystemSectionChange,
  };
}
