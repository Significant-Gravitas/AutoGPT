import { afterAll, beforeEach, describe, expect, it, vi } from "vitest";

// Read once at load by `flag-backend`, so it has to be set before the imports below.
const previousBackend = vi.hoisted(() => {
  const previous = process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND;
  process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND = "server";
  return previous;
});

import { getGetFeatureFlagsMockHandler } from "@/app/api/__generated__/endpoints/feature-flags/feature-flags.msw";
import type { FrontendFlagValues } from "@/app/api/__generated__/models/frontendFlagValues";
import openapi from "@/app/api/openapi.json";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { Flag, useFlagStatus } from "../use-get-flag";

vi.mock("@/app/(platform)/marketplace/components/HeroSection/helpers", () => ({
  DEFAULT_SEARCH_TERMS: [],
}));

const HIRE_EXPERTS = "hire-experts";

beforeEach(() => {
  signIn("u-1");
});

afterAll(() => {
  if (previousBackend === undefined) {
    delete process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND;
  } else {
    process.env.NEXT_PUBLIC_FEATURE_FLAG_BACKEND = previousBackend;
  }
  useAuthStore.setState({ user: null, isUserLoading: true });
});

describe("server backend", () => {
  it("serves the server's answer, and a flag it left out is not answered yet", async () => {
    serveFlags({ flags: { [HIRE_EXPERTS]: true }, configured: true });
    const { result } = renderHook(
      () => ({
        hire: useFlagStatus(Flag.HIRE_EXPERTS),
        search: useFlagStatus(Flag.CHAT_SEARCH),
      }),
      { wrapper: withQueryClient() },
    );

    await waitFor(() => expect(result.current.hire.answered).toBe(true));
    expect(result.current.hire.enabled).toBe(true);
    // A gated route must keep waiting rather than 404 on a flag the vendor never answered.
    expect(result.current.search).toMatchObject({
      enabled: false,
      answered: false,
    });
  });

  it("answers every flag with its default when the server has no vendor", async () => {
    serveFlags({ flags: {}, configured: false });

    const { result } = renderHook(() => useFlagStatus(Flag.CHAT_SEARCH), {
      wrapper: withQueryClient(),
    });

    await waitFor(() => expect(result.current.answered).toBe(true));
    expect(result.current.enabled).toBe(false);
  });

  it("waits for the session, then reads each signed-in user's own flags", async () => {
    const answers = [true, false];
    let requests = 0;
    serveFlags(() => ({
      flags: { [HIRE_EXPERTS]: answers[requests++] },
      configured: true,
    }));
    useAuthStore.setState({ isUserLoading: true });

    const { result } = renderHook(() => useFlagStatus(Flag.HIRE_EXPERTS), {
      wrapper: withQueryClient(),
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(requests).toBe(0);

    act(() => signIn("u-1"));
    await waitFor(() => expect(requests).toBe(1));
    await waitFor(() => expect(result.current.enabled).toBe(true));
    act(() => signIn("u-2"));
    await waitFor(() => expect(requests).toBe(2));
    await waitFor(() =>
      expect(result.current).toMatchObject({ enabled: false, answered: true }),
    );
  });
});

describe("the frontend's flags are the set the endpoint serves", () => {
  it("keys the endpoint's flags by the same enum the frontend reads from", () => {
    const schemas = openapi.components.schemas;
    const response =
      openapi.paths["/api/feature-flags"].get.responses["200"].content[
        "application/json"
      ].schema.$ref;

    expect(response).toBe("#/components/schemas/FrontendFlagValues");
    expect(schemas.FrontendFlagValues.properties.flags.propertyNames.$ref).toBe(
      "#/components/schemas/FrontendFlag",
    );
    expect([...schemas.FrontendFlag.enum].sort()).toEqual(
      Object.values(Flag).sort(),
    );
  });
});

function serveFlags(response: FrontendFlagValues | (() => FrontendFlagValues)) {
  server.use(getGetFeatureFlagsMockHandler(response));
}

function signIn(userID: string) {
  useAuthStore.setState({
    user: { id: userID, email: `${userID}@example.com`, user_metadata: {} },
    isUserLoading: false,
  });
}

function withQueryClient() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  };
}
