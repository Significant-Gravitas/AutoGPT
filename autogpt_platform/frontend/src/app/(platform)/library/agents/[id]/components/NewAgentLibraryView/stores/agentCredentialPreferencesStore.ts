import { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { storage } from "@/services/storage/local-storage";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

// Special marker to indicate "None" was explicitly selected
export const NONE_CREDENTIAL_MARKER = { __none__: true } as const;

type CredentialPreference =
  | CredentialsMetaInput
  | null
  | typeof NONE_CREDENTIAL_MARKER;

type AgentCredentialPreferences = Record<string, CredentialPreference>;

interface AgentCredentialPreferencesStore {
  preferences: Record<string, AgentCredentialPreferences>; // agentId -> preferences
  setCredentialPreference: (
    agentId: string,
    credentialKey: string,
    credential: CredentialPreference,
  ) => void;
  getCredentialPreference: (
    agentId: string,
    credentialKey: string,
  ) => CredentialPreference;
  clearPreference: (agentId: string, credentialKey: string) => void;
}

const STORAGE_KEY = "agent_credential_preferences";

// Custom storage adapter for localStorage
const customStorage = {
  getItem: (name: string): string | null => {
    return storage.get(name as any) || null;
  },
  setItem: (name: string, value: string): void => {
    storage.set(name as any, value);
  },
  removeItem: (name: string): void => {
    storage.clean(name as any);
  },
};

export const useAgentCredentialPreferencesStore =
  create<AgentCredentialPreferencesStore>()(
    persist(
      (set, get) => ({
        preferences: {},

        setCredentialPreference: (agentId, credentialKey, credential) => {
          set((state) => {
            const agentPrefs = state.preferences[agentId] || {};
            const updated = {
              ...state.preferences,
              [agentId]: {
                ...agentPrefs,
                [credentialKey]: credential,
              },
            };
            return { preferences: updated };
          });
        },

        getCredentialPreference: (agentId, credentialKey) => {
          const state = get();
          const pref = state.preferences[agentId]?.[credentialKey];
          // Convert serialized NONE marker back to constant
          if (
            pref &&
            typeof pref === "object" &&
            "__none__" in pref &&
            (pref as any).__none__ === true &&
            pref !== NONE_CREDENTIAL_MARKER
          ) {
            return NONE_CREDENTIAL_MARKER;
          }
          return pref ?? null;
        },

        clearPreference: (agentId, credentialKey) => {
          set((state) => {
            const agentPrefs = state.preferences[agentId] || {};
            const updated = { ...agentPrefs };
            delete updated[credentialKey];
            return {
              preferences: {
                ...state.preferences,
                [agentId]: updated,
              },
            };
          });
        },
      }),
      {
        name: STORAGE_KEY,
        storage: createJSONStorage(() => customStorage),
        // Transform on rehydrate to convert NONE markers
        onRehydrateStorage: () => (state, error) => {
          if (error || !state) {
            console.error("Failed to rehydrate credential preferences:", error);
            return;
          }
          // Convert serialized NONE markers back to constant
          const converted: Record<string, AgentCredentialPreferences> = {};
          for (const [agentId, prefs] of Object.entries(
            state.preferences || {},
          )) {
            const convertedPrefs: AgentCredentialPreferences = {};
            for (const [key, value] of Object.entries(prefs)) {
              if (
                value &&
                typeof value === "object" &&
                "__none__" in value &&
                (value as any).__none__ === true &&
                value !== NONE_CREDENTIAL_MARKER
              ) {
                convertedPrefs[key] = NONE_CREDENTIAL_MARKER;
              } else {
                convertedPrefs[key] = value as CredentialPreference;
              }
            }
            converted[agentId] = convertedPrefs;
          }
          // Update state with converted preferences
          if (
            Object.keys(converted).length > 0 ||
            Object.keys(state.preferences || {}).length > 0
          ) {
            state.preferences = converted;
          }
        },
      },
    ),
  );
