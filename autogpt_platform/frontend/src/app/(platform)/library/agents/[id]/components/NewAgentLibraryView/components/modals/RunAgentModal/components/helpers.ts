import { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { getSystemCredentials } from "../../CredentialsInputs/helpers";

export type CredentialField = [string, any];

type SavedCredential = {
  id: string;
  provider: string;
  type: string;
  title?: string | null;
};

function hasRequiredScopes(
  credential: { scopes?: string[]; type: string },
  requiredScopes?: string[],
) {
  if (credential.type !== "oauth2") return true;
  if (!requiredScopes || requiredScopes.length === 0) return true;
  const grantedScopes = new Set(credential.scopes || []);
  for (const scope of requiredScopes) {
    if (!grantedScopes.has(scope)) return false;
  }
  return true;
}

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
  requiredScopes: string[] | undefined,
  allProviders: CredentialsProvidersContextType | null,
): SavedCredential | undefined {
  for (const providerName of providerNames) {
    const providerData = allProviders?.[providerName];
    if (!providerData) continue;

    const systemCredentials = getSystemCredentials(
      providerData.savedCredentials ?? [],
    );

    const matchingCredentials: SavedCredential[] = [];

    for (const credential of systemCredentials) {
      if (
        credentialTypes.length > 0 &&
        !credentialTypes.includes(credential.type)
      )
        continue;
      if (!hasRequiredScopes(credential, requiredScopes)) continue;

      matchingCredentials.push(credential as SavedCredential);
    }

    if (matchingCredentials.length === 1) return matchingCredentials[0];
  }

  return undefined;
}
