"use client";

import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { storage } from "@/services/storage/local-storage";
import { useCallback, useEffect, useState } from "react";

// Special marker to indicate "None" was explicitly selected
export const NONE_CREDENTIAL_MARKER = { __none__: true } as const;

type AgentCredentialPreferences = Record<
  string,
  CredentialsMetaInput | null | typeof NONE_CREDENTIAL_MARKER
>;

const STORAGE_KEY_PREFIX = "agent_credential_prefs_";

function getStorageKey(agentId: string): string {
  return `${STORAGE_KEY_PREFIX}${agentId}`;
}

function loadPreferences(agentId: string): AgentCredentialPreferences {
  const key = getStorageKey(agentId);
  const stored = storage.get(key as any);
  if (!stored) return {};
  try {
    const parsed = JSON.parse(stored);
    // Convert serialized NONE markers back to the constant
    const result: AgentCredentialPreferences = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (
        value &&
        typeof value === "object" &&
        "__none__" in value &&
        (value as any).__none__ === true
      ) {
        result[key] = NONE_CREDENTIAL_MARKER;
      } else {
        result[key] = value as CredentialsMetaInput | null;
      }
    }
    return result;
  } catch {
    return {};
  }
}

function savePreferences(
  agentId: string,
  preferences: AgentCredentialPreferences,
): void {
  const key = getStorageKey(agentId);
  storage.set(key as any, JSON.stringify(preferences));
}

export function useAgentCredentialPreferences(agentId: string) {
  const [preferences, setPreferences] = useState<AgentCredentialPreferences>(
    () => loadPreferences(agentId),
  );

  useEffect(() => {
    const loaded = loadPreferences(agentId);
    setPreferences(loaded);
  }, [agentId]);

  const setCredentialPreference = useCallback(
    (
      credentialKey: string,
      credential: CredentialsMetaInput | null | typeof NONE_CREDENTIAL_MARKER,
    ) => {
      setPreferences((prev) => {
        const updated = {
          ...prev,
          [credentialKey]: credential,
        };
        savePreferences(agentId, updated);
        return updated;
      });
    },
    [agentId],
  );

  const getCredentialPreference = useCallback(
    (
      credentialKey: string,
    ): CredentialsMetaInput | null | typeof NONE_CREDENTIAL_MARKER => {
      return preferences[credentialKey] ?? null;
    },
    [preferences],
  );

  const clearPreference = useCallback(
    (credentialKey: string) => {
      setPreferences((prev) => {
        const updated = { ...prev };
        delete updated[credentialKey];
        savePreferences(agentId, updated);
        return updated;
      });
    },
    [agentId],
  );

  return {
    preferences,
    setCredentialPreference,
    getCredentialPreference,
    clearPreference,
  };
}
