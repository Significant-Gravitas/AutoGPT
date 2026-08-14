import { getListExpertIdentitiesQueryKey } from "@/app/api/__generated__/endpoints/experts/experts";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { useQueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { resolveExpertIdentity, useExpertMap } from "../../../useExpertMap";
import { ChatContainer } from "../ChatContainer";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    // Only HIRE_EXPERTS is on: useExpertMap runs, artifacts/task-bar stay off.
    useGetFlag: (flag: string) => flag === "hire-experts",
  };
});

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: ({ disabled }: { disabled?: boolean }) => (
    <button type="button" data-testid="composer" disabled={disabled} />
  ),
}));

vi.mock(
  "@/app/(platform)/copilot/components/ChatMessagesContainer/ChatMessagesContainer",
  () => ({
    ChatMessagesContainer: () => <div data-testid="messages" />,
  }),
);

vi.mock(
  "@/app/(platform)/copilot/components/ChatContainer/useAutoOpenArtifacts",
  () => ({
    useAutoOpenArtifacts: () => undefined,
  }),
);

vi.mock(
  "@/app/(platform)/copilot/components/UsageLimits/useIsUsageLimitReached",
  () => ({
    useIsUsageLimitReached: () => false,
  }),
);

const baseProps = {
  messages: [],
  status: "ready",
  error: undefined,
  sessionId: "s1",
  isLoadingSession: false,
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
  onStop: vi.fn(),
};

const maria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  tagline: null,
  bio: null,
  skills: [],
  identity: "You are Maria.",
  voice_preferences: "Warm.",
  boundaries: "Honest.",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

// The session already knows its expert; identity is resolved by the SAME
// endpoint→map→resolve chain production uses, so a fail-open regression here
// would surface as a writable thread rather than being masked by a hand-set prop.
function ExpertThreadHarness({ activeExpertId }: { activeExpertId: string }) {
  const queryClient = useQueryClient();
  const { expertsById, hasExpertsSettled } = useExpertMap();
  const expertIdentity = resolveExpertIdentity(
    activeExpertId,
    expertsById,
    hasExpertsSettled,
  );
  return (
    <>
      <button
        type="button"
        data-testid="refresh-identities"
        onClick={() =>
          void queryClient.invalidateQueries({
            queryKey: getListExpertIdentitiesQueryKey(),
          })
        }
      />
      <ChatContainer
        {...baseProps}
        expertIdentity={expertIdentity}
        isResolvingExpertIdentity={!hasExpertsSettled}
      />
    </>
  );
}

describe("Copilot expert thread — fail-closed identity", () => {
  it("keeps a fired expert's thread read-only with their real name", async () => {
    let requestedUrl: string | null = null;
    server.use(
      http.get("*/api/experts/identities", ({ request }) => {
        requestedUrl = request.url;
        return HttpResponse.json([{ ...maria, is_archived: true }]);
      }),
    );

    render(<ExpertThreadHarness activeExpertId="expert-maria" />);

    const notice = await screen.findByTestId("archived-expert-notice");
    expect(notice.textContent).toContain(
      "Maria was let go — this thread is read-only",
    );
    expect(screen.queryByTestId("composer")).toBeNull();
    expect(requestedUrl).toContain("/api/experts/identities");
  });

  it("fails closed to read-only when the roster fetch errors", async () => {
    server.use(
      http.get("*/api/experts/identities", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<ExpertThreadHarness activeExpertId="expert-maria" />);

    const notice = await screen.findByTestId("archived-expert-notice");
    expect(notice.textContent).toContain(
      "was let go — this thread is read-only",
    );
    expect(screen.queryByTestId("composer")).toBeNull();
  });

  it("fails closed to read-only when the expert is gone from a loaded roster", async () => {
    server.use(getListExpertIdentitiesMockHandler([]));

    render(<ExpertThreadHarness activeExpertId="expert-ghost" />);

    const notice = await screen.findByTestId("archived-expert-notice");
    expect(notice.textContent).toContain(
      "was let go — this thread is read-only",
    );
    expect(screen.queryByTestId("composer")).toBeNull();
  });

  it("keeps the composer for an active expert still on the roster", async () => {
    server.use(getListExpertIdentitiesMockHandler([maria]));

    render(<ExpertThreadHarness activeExpertId="expert-maria" />);

    await waitFor(() => expect(screen.getByTestId("composer")).toBeDefined());
    expect(screen.queryByTestId("archived-expert-notice")).toBeNull();
  });

  it("keeps the composer disabled until an expert identity resolves", async () => {
    let releaseRequest: () => void = () => undefined;
    const requestGate = new Promise<void>((resolve) => {
      releaseRequest = resolve;
    });
    server.use(
      http.get("*/api/experts/identities", async () => {
        await requestGate;
        return HttpResponse.json([maria]);
      }),
    );

    render(<ExpertThreadHarness activeExpertId="expert-maria" />);

    expect((screen.getByTestId("composer") as HTMLButtonElement).disabled).toBe(
      true,
    );

    act(() => releaseRequest());

    await waitFor(() =>
      expect(
        (screen.getByTestId("composer") as HTMLButtonElement).disabled,
      ).toBe(false),
    );
  });

  it("fails closed while cached identities refetch and if that refetch fails", async () => {
    let requestCount = 0;
    let releaseRefetch: () => void = () => undefined;
    const refetchGate = new Promise<void>((resolve) => {
      releaseRefetch = resolve;
    });
    server.use(
      http.get("*/api/experts/identities", async () => {
        requestCount += 1;
        if (requestCount === 1) return HttpResponse.json([maria]);
        await refetchGate;
        return HttpResponse.json({ detail: "boom" }, { status: 500 });
      }),
    );

    render(<ExpertThreadHarness activeExpertId="expert-maria" />);
    await waitFor(() =>
      expect(
        (screen.getByTestId("composer") as HTMLButtonElement).disabled,
      ).toBe(false),
    );

    act(() => screen.getByTestId("refresh-identities").click());
    await waitFor(() =>
      expect(
        (screen.getByTestId("composer") as HTMLButtonElement).disabled,
      ).toBe(true),
    );

    act(() => releaseRefetch());

    const notice = await screen.findByTestId("archived-expert-notice");
    expect(notice.textContent).toContain(
      "was let go — this thread is read-only",
    );
    expect(screen.queryByTestId("composer")).toBeNull();
  });
});
