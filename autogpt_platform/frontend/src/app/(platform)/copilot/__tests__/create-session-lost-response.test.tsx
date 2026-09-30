import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatInput } from "../components/ChatInput/ChatInput";
import { useChatSession } from "../useChatSession";

// A phone that loses the create response sees `TypeError: Load failed` even
// though the server committed the session. The composer and the create stay
// real here; only the periphery the composer pulls in is stubbed.

const testState = vi.hoisted(() => ({
  toast: vi.fn(),
}));

const hostedPlatform: ChatTransportResponse = {
  auth_provider: "platform",
  credential_id: null,
  label: "AutoGPT Platform",
  available: true,
  default: true,
};

vi.mock(
  "@/app/api/__generated__/endpoints/chat/chat",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/app/api/__generated__/endpoints/chat/chat")
      >();
    return {
      ...actual,
      useGetV2ListChatTransports: () => ({
        data: { status: 200, data: { transports: [hostedPlatform] } },
        isError: false,
      }),
      useGetV2ListChatConnections: () => ({
        data: { status: 200, data: { offers: [] } },
        isLoading: false,
        isPending: false,
        isError: false,
      }),
    };
  },
);

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => testState.toast(...args),
  useToast: () => ({ toast: testState.toast, dismiss: vi.fn() }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    CHAT_MODE_OPTION: "CHAT_MODE_OPTION",
    CHAT_WORKSPACE_FILES: "chat-workspace-files",
  },
  useGetFlag: () => false,
}));

vi.mock("../components/ChatInput/useVoiceRecording", () => ({
  useVoiceRecording: () => ({
    isRecording: false,
    isTranscribing: false,
    transcriptionError: null,
    hasFailedRecording: false,
    retryTranscription: vi.fn(),
    downloadFailedRecording: vi.fn(),
    dismissTranscriptionError: vi.fn(),
    elapsedTime: 0,
    toggleRecording: vi.fn(),
    handleKeyDown: vi.fn(),
    showMicButton: false,
    isInputDisabled: false,
    audioStream: null,
  }),
}));

vi.mock("@/components/ai-elements/prompt-input", () => ({
  PromptInputSubmit: ({ disabled }: { disabled?: boolean }) => (
    <button disabled={disabled} data-testid="submit">
      Send
    </button>
  ),
  PromptInputButton: () => null,
  PromptInputTextarea: (props: {
    id?: string;
    value?: string;
    onChange?: React.ChangeEventHandler<HTMLTextAreaElement>;
  }) => (
    <textarea
      id={props.id}
      value={props.value}
      onChange={props.onChange}
      data-testid="textarea"
    />
  ),
}));

function ComposerHarness() {
  const { createSession } = useChatSession();
  const [createdId, setCreatedId] = useState<string | null>(null);
  return (
    <>
      <ChatInput
        onSend={async () => {
          setCreatedId(await createSession());
        }}
      />
      <output data-testid="created-session">{createdId ?? ""}</output>
    </>
  );
}

function loseCreateResponses(lost: number) {
  const bodies: Array<{ session_id?: string }> = [];
  server.use(
    http.post("*/api/chat/sessions", async ({ request }) => {
      const body = ((await request.json()) ?? {}) as { session_id?: string };
      bodies.push(body);
      if (bodies.length <= lost) return HttpResponse.error();
      return HttpResponse.json({
        id: body.session_id ?? "server-minted",
        created_at: "2026-09-30T11:03:02Z",
        user_id: "user-1",
      });
    }),
  );
  return bodies;
}

function composerValue() {
  return (screen.getByTestId("textarea") as HTMLTextAreaElement).value;
}

function createdSession() {
  return screen.getByTestId("created-session").textContent;
}

function send(text: string) {
  fireEvent.change(screen.getByTestId("textarea"), {
    target: { value: text },
  });
  fireEvent.click(screen.getByTestId("submit"));
}

afterEach(() => {
  server.resetHandlers();
  testState.toast.mockClear();
});

describe("creating a session when the response is lost", () => {
  it("retries once with the same session id and keeps the message sent", async () => {
    const bodies = loseCreateResponses(1);
    render(<ComposerHarness />);

    send("Hi Anika");

    await waitFor(() => expect(bodies).toHaveLength(2), { timeout: 3000 });
    expect(bodies[0].session_id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/,
    );
    expect(bodies[1].session_id).toBe(bodies[0].session_id);
    await waitFor(() => expect(createdSession()).toBe(bodies[0].session_id));
    expect(composerValue()).toBe("");
    expect(testState.toast).not.toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("gives the message back after the one retry also fails", async () => {
    const bodies = loseCreateResponses(2);
    render(<ComposerHarness />);

    send("Hi Anika");

    await waitFor(() => expect(composerValue()).toBe("Hi Anika"), {
      timeout: 3000,
    });
    expect(bodies).toHaveLength(2);
    expect(bodies[1].session_id).toBe(bodies[0].session_id);
  });
});
