import type { ExpertIdentity } from "@/app/api/__generated__/models/expertIdentity";
import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useExpertMap } from "../../../useExpertMap";
import { useRecipientPicker } from "../useRecipientPicker";

const MARIA: ExpertIdentity = {
  id: "expert-maria",
  name: "Maria",
  role: "Email & Lifecycle",
  avatar_url: null,
  is_archived: false,
};

let client: QueryClient;
let release: () => void = () => {};

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});

afterEach(() => {
  release();
});

function Wrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider client={client}>
      <NuqsTestingAdapter>{children}</NuqsTestingAdapter>
    </QueryClientProvider>
  );
}

function renderPicker() {
  return renderHook(
    () => ({ picker: useRecipientPicker(), roster: useExpertMap() }),
    { wrapper: Wrapper },
  );
}

describe("useRecipientPicker — a team that can't be read", () => {
  it("counts the team as unreadable until a read succeeds, a retry in flight included", async () => {
    let reads = 0;
    const retry = new Promise<void>((resolve) => {
      release = resolve;
    });
    server.use(
      http.get("*/api/internal/experts/identities", async () => {
        reads += 1;
        if (reads === 1) return HttpResponse.json([MARIA]);
        if (reads === 2) {
          return HttpResponse.json({ detail: "down" }, { status: 500 });
        }
        await retry;
        return HttpResponse.json([MARIA]);
      }),
    );
    const { result } = renderPicker();
    await waitFor(() => expect(result.current.picker.experts).toHaveLength(1));
    expect(result.current.picker.hasRosterFailed).toBe(false);

    await act(() => client.refetchQueries());
    await waitFor(() =>
      expect(result.current.picker.hasRosterFailed).toBe(true),
    );
    expect(result.current.picker.experts).toEqual([]);

    let refetch!: Promise<void>;
    act(() => {
      refetch = client.refetchQueries();
    });
    await waitFor(() =>
      expect(result.current.roster.isLoadingExperts).toBe(true),
    );
    expect(result.current.picker.hasRosterFailed).toBe(true);

    release();
    await act(() => refetch);
    await waitFor(() =>
      expect(result.current.picker.hasRosterFailed).toBe(false),
    );
    expect(result.current.picker.experts).toHaveLength(1);
  });
});
