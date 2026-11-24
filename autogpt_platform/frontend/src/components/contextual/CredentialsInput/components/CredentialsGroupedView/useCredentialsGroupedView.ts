import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
  CredentialsType,
} from "@/lib/autogpt-server-api/types";
import {
  CredentialsProvidersContext,
  CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import { useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  areSystemCredentialProvidersLoading,
  CredentialField,
  findSavedCredentialByProviderAndType,
  hasMissingRequiredSystemCredentials,
  splitCredentialFieldsBySystem,
} from "./helpers";
import {
  awaitsUserChoice,
  CredentialFieldSchema,
  findOwnCredentialProvider,
  getNodeCredentialsKey,
  getStartingCredential,
  isOwnCredential,
} from "./node-credentials";

interface Args {
  credentialFields: CredentialField[];
  requiredCredentials: Set<string>;
  inputCredentials: Record<string, CredentialsMetaInput | undefined>;
  onCredentialChange: (key: string, value?: CredentialsMetaInput) => void;
  onAutoSelect?: (key: string, value: CredentialsMetaInput) => void;
  nodeCredentials: CredentialsMetaInput[];
  nodeCredentialsUnknown: boolean;
  clearedFields: string[];
}

/** A field with the schema its picker gets; none when no provider resolves. */
export interface PickerField {
  key: string;
  schema: BlockIOCredentialsSubSchema;
  pickerSchema: BlockIOCredentialsSubSchema | null;
}

export function useCredentialsGroupedView({
  credentialFields,
  requiredCredentials,
  inputCredentials,
  onCredentialChange,
  onAutoSelect,
  nodeCredentials,
  nodeCredentialsUnknown,
  clearedFields,
}: Args) {
  const allProviders = useContext(CredentialsProvidersContext);

  const { userCredentialFields, systemCredentialFields } = useMemo(
    () => splitCredentialFieldsBySystem(credentialFields, allProviders),
    [credentialFields, allProviders],
  );

  const hasSystemCredentials = systemCredentialFields.length > 0;
  // Empty fields are filled once for each set of step keys they could start on
  const autoSelectBasis = `${getNodeCredentialsKey(nodeCredentials)}|${nodeCredentialsUnknown}`;
  const autoSelectedFor = useRef<string | null>(null);
  // A field the user changes stays where it is: one shown above the system
  // credentials keeps its place if they switch it to the platform key, and
  // one changed inside the open section waits there until it closes.
  const [pinnedAbove, setPinnedAbove] = useState<string[]>([]);
  const [pinnedInside, setPinnedInside] = useState<string[]>([]);

  // A field set to one of the user's own keys is shown above the collapsed
  // system credentials, so they can see which key the run will use, and so
  // is one left for them to pick, or a required one they have to add a key for.
  function isShownAbove([key, schema]: CredentialField) {
    if (pinnedAbove.includes(key)) return true;
    if (pinnedInside.includes(key)) return false;
    if (isOwnCredential(inputCredentials?.[key], allProviders)) return true;
    if (inputCredentials?.[key]) return false;
    const isRequired = requiredCredentials.has(key);
    if (isRequired && !getPickerSchema(schema, undefined, allProviders)) {
      return true;
    }
    return awaitsUserChoice({
      schema,
      isRequired,
      nodeCredentialsUnknown,
      allProviders,
    });
  }

  const promotedFields = systemCredentialFields.filter(isShownAbove);
  const collapsedFields = systemCredentialFields.filter(
    (field) => !isShownAbove(field),
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

  function withPickerSchemas(fields: CredentialField[]): PickerField[] {
    return fields.map(([key, schema]) => ({
      key,
      schema,
      pickerSchema: getPickerSchema(
        schema,
        inputCredentials?.[key],
        allProviders,
      ),
    }));
  }

  function handleFieldChange(key: string, value?: CredentialsMetaInput) {
    const systemField = systemCredentialFields.find(
      ([systemKey]) => systemKey === key,
    );
    if (systemField) {
      const pin = isShownAbove(systemField) ? setPinnedAbove : setPinnedInside;
      pin((keys) => (keys.includes(key) ? keys : [...keys, key]));
    }
    onCredentialChange(key, value);
  }

  function handleSystemSectionChange(openSection: string) {
    if (!openSection) setPinnedInside([]);
  }

  useEffect(() => {
    if (autoSelectedFor.current === autoSelectBasis) return;
    if (!hasSystemCredentials && nodeCredentials.length === 0) return;
    if (!allProviders || isLoadingProviders) return;

    const select = onAutoSelect ?? onCredentialChange;
    for (const [key, schema] of credentialFields) {
      // A field the user cleared stays empty: "None" is their choice
      if (inputCredentials?.[key] || clearedFields.includes(key)) continue;

      const isRequired = requiredCredentials.has(key);
      const isRequiredSystemField =
        isRequired &&
        systemCredentialFields.some(([systemKey]) => systemKey === key);
      const credential = getStartingCredential({
        schema,
        isRequired,
        nodeCredentials,
        nodeCredentialsUnknown,
        allProviders,
        findFallbackCredential: () =>
          isRequiredSystemField
            ? findFallbackCredential(schema, allProviders)
            : undefined,
      });
      if (credential) select(key, credential);
    }

    autoSelectedFor.current = autoSelectBasis;
  }, [
    allProviders,
    autoSelectBasis,
    clearedFields,
    credentialFields,
    hasSystemCredentials,
    inputCredentials,
    isLoadingProviders,
    nodeCredentials,
    nodeCredentialsUnknown,
    onAutoSelect,
    onCredentialChange,
    requiredCredentials,
    systemCredentialFields,
  ]);

  return {
    visibleFields: withPickerSchemas([
      ...userCredentialFields,
      ...promotedFields,
    ]),
    collapsedFields: withPickerSchemas(collapsedFields),
    hasMissingSystemCredentials,
    handleFieldChange,
    handleSystemSectionChange,
  };
}

// A picker shows one provider's keys. A field open to several, like an LLM
// step's whose model comes from a link, shows the provider of the key it
// holds, else the first one the user has a key of their own for; with
// neither it has no picker. The discriminator is left out: the picker goes by
// provider, not by the step's other inputs.
function getPickerSchema<Schema extends CredentialFieldSchema>(
  schema: Schema,
  selected: CredentialsMetaInput | undefined,
  allProviders: CredentialsProvidersContextType | null,
): Schema | null {
  const pickerSchema = { ...schema, discriminator: undefined };
  const providers = schema.credentials_provider ?? [];
  if (providers.length <= 1) return pickerSchema;

  const provider =
    selected && providers.includes(selected.provider)
      ? selected.provider
      : findOwnCredentialProvider(schema, allProviders);
  return provider
    ? { ...pickerSchema, credentials_provider: [provider] }
    : null;
}

// The platform's key, or failing that the only key that fits
function findFallbackCredential(
  schema: CredentialFieldSchema,
  allProviders: CredentialsProvidersContextType,
): CredentialsMetaInput | undefined {
  const saved = findSavedCredentialByProviderAndType(
    schema.credentials_provider || [],
    schema.credentials_types || [],
    schema.credentials_scopes,
    allProviders,
    schema.discriminator_values,
  );
  if (!saved) return undefined;
  return {
    id: saved.id,
    provider: saved.provider,
    type: saved.type as CredentialsType,
    title: saved.title,
  };
}
