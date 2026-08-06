import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

// Recommendations only exist behind the brain-dump flag; these tests are
// about what the panel does once it is on.
vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === actual.Flag.ONBOARDING_BRAIN_DUMP,
  };
});

import { useConnectToolsPanel } from "../useConnectToolsPanel";

const PROVIDERS_URL =
  "http://localhost:3000/api/proxy/api/integrations/providers";
const CREDENTIALS_URL =
  "http://localhost:3000/api/proxy/api/integrations/credentials";
const RECOMMENDED_URL =
  "http://localhost:3000/api/proxy/api/onboarding/brain-dump/recommended-providers";

function stub(recommendations: { provider: string; reason?: string }[]) {
  server.use(
    http.get(PROVIDERS_URL, () =>
      HttpResponse.json([
        {
          name: "notion",
          description: "Docs and wikis",
          supported_auth_types: ["api_key"],
        },
        {
          name: "slack",
          description: "Team chat",
          supported_auth_types: ["oauth2"],
        },
      ]),
    ),
    http.get(CREDENTIALS_URL, () => HttpResponse.json([])),
    http.get(RECOMMENDED_URL, () =>
      HttpResponse.json({ ready: true, providers: recommendations }),
    ),
  );
}

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function setup() {
  return renderHook(() => useConnectToolsPanel(), { wrapper });
}

describe("useConnectToolsPanel — recommendation mapping", () => {
  it("swaps the registry blurb for the model's reason", async () => {
    stub([{ provider: "notion", reason: "You said your notes live there" }]);

    const { result } = setup();

    await waitFor(() =>
      expect(result.current.recommendedProviders).toHaveLength(1),
    );
    expect(result.current.recommendedProviders[0]).toMatchObject({
      id: "notion",
      name: "Notion",
      description: "You said your notes live there",
    });
  });

  it("keeps the registry blurb when the model returns no reason", async () => {
    stub([{ provider: "notion", reason: "" }]);

    const { result } = setup();

    await waitFor(() =>
      expect(result.current.recommendedProviders).toHaveLength(1),
    );
    expect(result.current.recommendedProviders[0].description).toBe(
      "Docs and wikis",
    );
  });

  it("drops picks whose provider is not in the live registry", async () => {
    stub([
      { provider: "retired_tool", reason: "Gone since the job ran" },
      { provider: "slack", reason: "Your team lives in chat" },
    ]);

    const { result } = setup();

    await waitFor(() =>
      expect(result.current.recommendedProviders).toHaveLength(1),
    );
    expect(result.current.recommendedProviders[0].id).toBe("slack");
  });
});

describe("useConnectToolsPanel — selection", () => {
  it("clears a half-typed API key when a different provider is picked", async () => {
    stub([{ provider: "notion" }, { provider: "slack" }]);

    const { result } = setup();
    await waitFor(() => expect(result.current.providers).toHaveLength(2));

    act(() => result.current.handleSelect("notion"));
    act(() =>
      result.current.apiKeyForm.setValue("apiKey", "sk-half-typed", {
        shouldValidate: true,
      }),
    );
    expect(result.current.apiKeyForm.getValues("apiKey")).toBe("sk-half-typed");

    act(() => result.current.handleSelect("slack"));

    expect(result.current.selectedProvider?.id).toBe("slack");
    expect(result.current.selectedMethod).toBeNull();
    // A key typed for the previous provider must not carry over.
    expect(result.current.apiKeyForm.getValues("apiKey")).toBe("");
  });

  it("returns to the list and forgets the chosen method on back", async () => {
    stub([{ provider: "slack" }]);

    const { result } = setup();
    await waitFor(() => expect(result.current.providers).toHaveLength(2));

    act(() => result.current.handleSelect("slack"));
    act(() => result.current.setSelectedMethod("oauth2"));
    expect(result.current.isContinueDisabled).toBe(false);

    act(() => result.current.handleBackToList());

    expect(result.current.selectedProvider).toBeNull();
    expect(result.current.selectedMethod).toBeNull();
    expect(result.current.isContinueDisabled).toBe(true);
    expect(result.current.direction).toBe(-1);
  });

  it("hides Continue for methods it cannot drive", async () => {
    stub([]);

    const { result } = setup();
    await waitFor(() => expect(result.current.providers).toHaveLength(2));

    act(() => result.current.handleSelect("notion"));
    act(() => result.current.setSelectedMethod("host_scoped"));

    expect(result.current.showContinue).toBe(false);

    act(() => result.current.setSelectedMethod("api_key"));

    expect(result.current.showContinue).toBe(true);
    // An empty key form is not submittable.
    expect(result.current.isContinueDisabled).toBe(true);
  });
});
