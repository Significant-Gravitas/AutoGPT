import { server } from "@/mocks/mock-server";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { UIMessage } from "ai";
import { http, HttpResponse } from "msw";
import { parseAsString, useQueryState } from "nuqs";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { buildKickoffMessage, markKickedOff } from "../expertKickoff";
import { useCopilotUIStore } from "../store";
import { useChatSession } from "../useChatSession";
import { useExpertKickoff } from "../useExpertKickoff";
import { useSendMessage } from "../useSendMessage";

const flagState = vi.hoisted(() => ({
  values: { "hire-experts": true } as Record<string, boolean>,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flagState.values[flag] ?? false,
  };
});

const sendSpy = vi.hoisted(() => vi.fn());

afterEach(() => {
  cleanup();
  server.resetHandlers();
  sendSpy.mockClear();
  flagState.values = { "hire-experts": true };
  window.localStorage.clear();
  useCopilotUIStore.setState({ adoptedExpertThreads: new Set<string>() });
  useCopilotStreamStore.getState().resetAll();
});

function KickoffHarness() {
  const [expertId] = useQueryState("expertId", parseAsString);
  const [kickoff] = useQueryState("kickoff", parseAsString);
  const { sessionId, createSession } = useChatSession({ expertId });
  const isUserStoppingRef = { current: false };
  const { onSend } = useSendMessage({
    sessionId,
    sendMessage: sendSpy as never,
    createSession,
    isUserStoppingRef,
  });
  useExpertKickoff({
    expertId,
    kickoff: kickoff === "1",
    sessionId,
    onKickoff: () => void onSend(buildKickoffMessage()),
  });
  return <div data-testid="session-id">{sessionId ?? "none"}</div>;
}

function renderKickoff(searchParams: string) {
  const Wrapper = withNuqsTestingAdapter({ searchParams, hasMemory: true });
  return render(
    <CredentialsProvidersContext.Provider value={{}}>
      <Wrapper>
        <KickoffHarness />
      </Wrapper>
    </CredentialsProvidersContext.Provider>,
  );
}

function stubTransports() {
  return http.get("*/api/chat/transports", () =>
    HttpResponse.json({
      transports: [
        {
          auth_provider: "platform",
          credential_id: null,
          label: "AutoGPT Platform",
          available: true,
          default: true,
        },
      ],
    }),
  );
}

describe("useExpertKickoff", () => {
  it("creates an expert session and sends the kickoff message exactly once", async () => {
    let createBody: unknown = null;
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({ sessions: [], total: 0 }),
      ),
      http.post("*/api/chat/sessions", async ({ request }) => {
        createCount += 1;
        createBody = await request.json();
        return HttpResponse.json({
          id: "kickoff-session-1",
          created_at: "2026-01-01T00:00:00Z",
          user_id: "user-1",
          expert_id: "expert-maria",
        });
      }),
      http.get("*/api/chat/sessions/kickoff-session-1", () =>
        HttpResponse.json({
          id: "kickoff-session-1",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-01T00:00:00Z",
          user_id: "user-1",
          expert_id: "expert-maria",
          messages: [],
        }),
      ),
    );

    renderKickoff("?expertId=expert-maria&kickoff=1");

    await waitFor(() => expect(createCount).toBe(1));
    expect(createBody).toEqual({
      expert_id: "expert-maria",
      llm_auth_provider: "platform",
    });

    await waitFor(() => expect(sendSpy).toHaveBeenCalledTimes(1));
    const firstArg = sendSpy.mock.calls[0][0] as { text: string };
    expect(firstArg.text).toBe(buildKickoffMessage());

    // The message is never re-sent even after everything settles.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sendSpy).toHaveBeenCalledTimes(1);
    expect(createCount).toBe(1);
  });

  it("no-ops on a second visit once the expert has been kicked off", async () => {
    markKickedOff("expert-maria");
    let createCount = 0;
    server.use(
      stubTransports(),
      // The list query here is `useChatSession`'s adoption lookup, not the
      // kickoff's — returning empty keeps the user on the new-task screen.
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({ sessions: [], total: 0 }),
      ),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff("?expertId=expert-maria&kickoff=1");

    await new Promise((resolve) => setTimeout(resolve, 80));
    // The localStorage latch keeps the kickoff from creating or sending again.
    expect(sendSpy).not.toHaveBeenCalled();
    expect(createCount).toBe(0);
    expect(screen.getByTestId("session-id").textContent).toBe("none");
  });

  it("adopts the existing thread and skips the kickoff when one already exists", async () => {
    let createCount = 0;
    server.use(
      stubTransports(),
      http.get("*/api/chat/sessions", () =>
        HttpResponse.json({
          sessions: [
            {
              id: "existing-maria",
              title: "Maria thread",
              expert_id: "expert-maria",
              is_processing: false,
              created_at: "2026-01-01T00:00:00Z",
              updated_at: "2026-01-02T00:00:00Z",
            },
          ],
          total: 1,
        }),
      ),
      http.get("*/api/chat/sessions/existing-maria", () =>
        HttpResponse.json({
          id: "existing-maria",
          created_at: "2026-01-01T00:00:00Z",
          updated_at: "2026-01-02T00:00:00Z",
          user_id: "user-1",
          expert_id: "expert-maria",
          messages: [] as UIMessage[],
        }),
      ),
      http.post("*/api/chat/sessions", () => {
        createCount += 1;
        return HttpResponse.json({ id: "should-not-happen" });
      }),
    );

    renderKickoff("?expertId=expert-maria&kickoff=1");

    await waitFor(() =>
      expect(screen.getByTestId("session-id").textContent).toBe(
        "existing-maria",
      ),
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(sendSpy).not.toHaveBeenCalled();
    expect(createCount).toBe(0);
  });
});
