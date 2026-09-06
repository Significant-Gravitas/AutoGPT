import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import React from "react";
import {
  CredentialsProvidersContext,
  type CredentialsProvidersContextType,
} from "@/providers/agent-credentials/credentials-provider";
import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api";
import useCredentials from "../useCredentials";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

const schema: BlockIOCredentialsSubSchema = {
  type: "object",
  properties: {},
  credentials_provider: ["github"],
  credentials_types: ["oauth2"],
};

function makeProviders(...names: string[]): CredentialsProvidersContextType {
  return Object.fromEntries(
    names.map((name) => [
      name,
      {
        provider: name,
        providerName: name,
        savedCredentials: [],
        isSystemProvider: false,
      },
    ]),
  ) as unknown as CredentialsProvidersContextType;
}

function renderWithProviders(
  providers: CredentialsProvidersContextType | null,
) {
  return renderHook(() => useCredentials(schema), {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <CredentialsProvidersContext.Provider value={providers}>
        {children}
      </CredentialsProvidersContext.Provider>
    ),
  });
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("a provider the frontend has never heard of", () => {
  it("counts the card that will never render", async () => {
    const { result } = renderWithProviders(makeProviders("google"));

    // The null is what the card reads as "still loading", forever.
    expect(result.current).toBeNull();
    await waitFor(() =>
      expect(capture).toHaveBeenCalledWith("credential_card_never_rendered", {
        failure_class: "class_03_provider_unknown_to_frontend",
        provider: "github",
      }),
    );
  });

  it("counts nothing while the provider map is still loading", async () => {
    const { result } = renderWithProviders(null);

    expect(result.current).toBeNull();
    await waitFor(() => expect(capture).not.toHaveBeenCalled());
  });

  it("counts nothing once the provider is in the map", async () => {
    const { result } = renderWithProviders(makeProviders("github"));

    expect(result.current).not.toBeNull();
    await waitFor(() => expect(capture).not.toHaveBeenCalled());
  });
});
