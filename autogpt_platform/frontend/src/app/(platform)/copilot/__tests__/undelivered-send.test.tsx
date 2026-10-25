import { server } from "@/mocks/mock-server";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotUIStore } from "../store";
import { getTurnRuntime } from "../stream/turnRuntime";
import {
  renderHost,
  resetChatRuntimes,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "./sse-helpers";

// On the stream runtime, a send whose stream the server closed with nothing
// taking it (no turn, no queue, no row) is said not to have gone through, and
// its text is offered back without overwriting what the user is typing.

const toast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => toast(...args),
  useToast: () => ({ toast, dismiss: vi.fn() }),
}));

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("../helpers", async (importActual) => {
  const actual = await importActual<typeof import("../helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === "copilot-stream-runtime",
  };
});

const SENT = "What is my team working on?";

beforeEach(() => {
  resetChatRuntimes();
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState({ initialPrompt: null });
  toast.mockReset();
});

afterEach(() => {
  resetChatRuntimes();
  server.resetHandlers();
});

function answerWithABareStream() {
  server.use(
    http.post(
      `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`,
      () =>
        new HttpResponse(
          [
            { type: "start", messageId: "nowhere" },
            { type: "start-step" },
            { type: "finish-step" },
            { type: "finish" },
          ]
            .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
            .concat("data: [DONE]\n\n")
            .join(""),
          { headers: { "content-type": "text/event-stream" } },
        ),
    ),
  );
}

function composer() {
  return screen.getByLabelText(/chat message input/i) as HTMLTextAreaElement;
}

describe("a send that went nowhere", () => {
  it("says so and puts the text back in an empty composer", async () => {
    renderHost();
    answerWithABareStream();

    await typeAndSend(SENT);

    expect(
      await screen.findByText(
        "Your message didn't go through. Send it again.",
        undefined,
        {
          timeout: 10_000,
        },
      ),
    ).toBeDefined();
    await waitFor(() => expect(composer().value).toBe(SENT));
  }, 20_000);

  it("names the files of a lost send that was only files, instead of dropping it silently", async () => {
    renderHost();
    answerWithABareStream();
    await waitFor(() => expect(getTurnRuntime(TEST_SESSION_ID)).toBeDefined());

    void getTurnRuntime(TEST_SESSION_ID)!.send(
      {
        parts: [
          {
            type: "file",
            url: "/api/proxy/api/workspace/files/file-1/download",
            mediaType: "application/pdf",
            filename: "q3-plan.pdf",
          },
        ],
      },
      undefined,
    );

    expect(
      await screen.findByText(
        "Your message didn't go through. Attach your files again and send it.",
        undefined,
        { timeout: 10_000 },
      ),
    ).toBeDefined();
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Your files weren't sent",
        description: "Attach q3-plan.pdf again and send.",
      }),
    );
    expect(composer().value).toBe("");
  }, 20_000);

  it("leaves a new draft alone and still offers the lost text", async () => {
    renderHost();
    answerWithABareStream();

    await typeAndSend(SENT);
    // The user starts the next message while the lost one is being recovered.
    await userEvent.setup().type(composer(), "And Monday?");

    await waitFor(
      () =>
        expect(toast).toHaveBeenCalledWith(
          expect.objectContaining({ description: SENT }),
        ),
      { timeout: 10_000 },
    );
    expect(composer().value).toBe("And Monday?");
    expect(
      useCopilotStreamStore.getState().getCoord(TEST_SESSION_ID)
        .lastSubmittedMessageText,
    ).toBe(SENT);
  }, 20_000);
});
