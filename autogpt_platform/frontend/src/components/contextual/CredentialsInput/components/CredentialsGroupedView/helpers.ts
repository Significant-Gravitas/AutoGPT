import { CredentialsProvidersContextType } from "@/providers/agent-credentials/credentials-provider";
import { filterSystemCredentials, getSystemCredentials } from "../../helpers";

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

export function areSystemCredentialProvidersLoading(
  systemCredentialFields: CredentialField[],
  allProviders: CredentialsProvidersContextType | null,
): boolean {
  if (!systemCredentialFields.length) return false;
  if (allProviders === null) return true;

  for (const [_, schema] of systemCredentialFields) {
    const providerNames = schema.credentials_provider || [];
    const hasAllProviders = providerNames.every(
      (providerName: string) => allProviders?.[providerName] !== undefined,
    );
    if (!hasAllProviders) return true;
  }

  return false;
}

export function hasMissingRequiredSystemCredentials(
  systemCredentialFields: CredentialField[],
  requiredCredentials: Set<string>,
  inputCredentials?: Record<string, unknown>,
  allProviders?: CredentialsProvidersContextType | null,
) {
  if (systemCredentialFields.length === 0) return false;
  if (allProviders === null) return false;

  return systemCredentialFields.some(([key, schema]) => {
    if (!requiredCredentials.has(key)) return false;
    if (inputCredentials?.[key]) return false;

    const providerNames = schema.credentials_provider || [];
    const credentialTypes = schema.credentials_types || [];
    const requiredScopes = schema.credentials_scopes;

    return !hasAvailableSystemCredential(
      providerNames,
      credentialTypes,
      requiredScopes,
      allProviders,
    );
  });
}

function hasAvailableSystemCredential(
  providerNames: string[],
  credentialTypes: string[],
  requiredScopes: string[] | undefined,
  allProviders: CredentialsProvidersContextType | null | undefined,
) {
  if (!allProviders) return false;

  for (const providerName of providerNames) {
    const providerData = allProviders[providerName];
    if (!providerData) continue;

    const systemCredentials = getSystemCredentials(
      providerData.savedCredentials ?? [],
    );

    for (const credential of systemCredentials) {
      const typeMatches =
        credentialTypes.length === 0 ||
        credentialTypes.includes(credential.type);
      const scopesMatch = hasRequiredScopes(credential, requiredScopes);

      if (!typeMatches) continue;
      if (!scopesMatch) continue;

      return true;
    }

    const allCredentials = providerData.savedCredentials ?? [];
    for (const credential of allCredentials) {
      const typeMatches =
        credentialTypes.length === 0 ||
        credentialTypes.includes(credential.type);
      const scopesMatch = hasRequiredScopes(credential, requiredScopes);

      if (!typeMatches) continue;
      if (!scopesMatch) continue;

      return true;
    }
  }

  return false;
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
      const typeMatches =
        credentialTypes.length === 0 ||
        credentialTypes.includes(credential.type);
      const scopesMatch = hasRequiredScopes(credential, requiredScopes);

      if (!typeMatches) continue;
      if (!scopesMatch) continue;

      matchingCredentials.push(credential as SavedCredential);
    }

    if (matchingCredentials.length === 0) {
      const allCredentials = providerData.savedCredentials ?? [];
      for (const credential of allCredentials) {
        const typeMatches =
          credentialTypes.length === 0 ||
          credentialTypes.includes(credential.type);
        const scopesMatch = hasRequiredScopes(credential, requiredScopes);

        if (!typeMatches) continue;
        if (!scopesMatch) continue;

        matchingCredentials.push(credential as SavedCredential);
      }
    }

    if (matchingCredentials.length === 1) {
      return matchingCredentials[0];
    }
    if (matchingCredentials.length > 1) {
      return undefined;
    }
  }

  return undefined;
}

export function findSavedUserCredentialByProviderAndType(
  providerNames: string[],
  credentialTypes: string[],
  requiredScopes: string[] | undefined,
  allProviders: CredentialsProvidersContextType | null,
): SavedCredential | undefined {
  for (const providerName of providerNames) {
    const providerData = allProviders?.[providerName];
    if (!providerData) continue;

    const userCredentials = filterSystemCredentials(
      providerData.savedCredentials ?? [],
    );

    const matchingCredentials: SavedCredential[] = [];

    for (const credential of userCredentials) {
      const typeMatches =
        credentialTypes.length === 0 ||
        credentialTypes.includes(credential.type);
      const scopesMatch = hasRequiredScopes(credential, requiredScopes);

      if (!typeMatches) continue;
      if (!scopesMatch) continue;

      matchingCredentials.push(credential as SavedCredential);
    }

    if (matchingCredentials.length === 1) {
      return matchingCredentials[0];
    }
    if (matchingCredentials.length > 1) {
      return undefined;
    }
  }

  return undefined;
}
