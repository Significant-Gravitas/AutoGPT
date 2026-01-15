import { getSystemCredentials } from "../../CredentialsInputs/helpers";
import { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";

export type CredentialField = [string, any];

type SavedCredential = {
  id: string;
  provider: string;
  type: string;
  title?: string | null;
};

export function splitCredentialFieldsBySystem(
  credentialFields: CredentialField[],
  allProviders: CredentialsProvidersContextType | null,
  inputCredentials?: Record<string, unknown>,
) {
  if (!allProviders || credentialFields.length === 0) {
    return {
      userCredentialFields: [] as CredentialField[],
      systemCredentialFields: [] as CredentialField[],
    };
  }

  const userFields: CredentialField[] = [];
  const systemFields: CredentialField[] = [];

  for (const [key, schema] of credentialFields) {
    const providerNames = schema.credentials_provider || [];
    const isSystemField = providerNames.some((providerName: string) => {
      const providerData = allProviders[providerName];
      return providerData?.isSystemProvider === true;
    });

    if (isSystemField) {
      systemFields.push([key, schema]);
    } else {
      userFields.push([key, schema]);
    }
  }

  const sortByUnsetFirst = (a: CredentialField, b: CredentialField) => {
    const aIsSet = Boolean(inputCredentials?.[a[0]]);
    const bIsSet = Boolean(inputCredentials?.[b[0]]);

    if (aIsSet === bIsSet) return 0;
    return aIsSet ? 1 : -1;
  };

  return {
    userCredentialFields: userFields.sort(sortByUnsetFirst),
    systemCredentialFields: systemFields.sort(sortByUnsetFirst),
  };
}

export function hasMissingRequiredSystemCredentials(
  systemCredentialFields: CredentialField[],
  requiredCredentials: Set<string>,
  inputCredentials?: Record<string, unknown>,
) {
  if (systemCredentialFields.length === 0) return false;

  return systemCredentialFields.some(([key]) => {
    const isRequired = requiredCredentials.has(key);
    const selectedCred = inputCredentials?.[key];
    return isRequired && !selectedCred;
  });
}

export function findSavedCredentialByProviderAndType(
  providerNames: string[],
  credentialTypes: string[],
  allProviders: CredentialsProvidersContextType | null,
): SavedCredential | undefined {
  for (const providerName of providerNames) {
    const providerData = allProviders?.[providerName];
    if (!providerData) continue;

    const systemCredentials = getSystemCredentials(
      providerData.savedCredentials ?? [],
    );

    const matchingCredential = systemCredentials.find((credential) => {
      if (credentialTypes.length > 0) {
        return credentialTypes.includes(credential.type);
      }
      return true;
    });

    if (matchingCredential) {
      return matchingCredential as SavedCredential;
    }
  }

  return undefined;
}
