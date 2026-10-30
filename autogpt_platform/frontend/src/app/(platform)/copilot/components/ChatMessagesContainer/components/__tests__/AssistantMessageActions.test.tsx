import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";
import type { UIDataTypes, UIMessage, UITools } from "ai";

import { getPostV2SubmitMessageFeedbackMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { MessageFeedbackRequest } from "@/app/api/__generated__/models/messageFeedbackRequest";
import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { AssistantMessageActions } from "../AssistantMessageActions";

const { toastMock, captureExceptionMock } = vi.hoisted(() => ({
  toastMock: vi.fn(),
  captureExceptionMock: vi.fn(),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: toastMock,
  useToast: () => ({ toast: toastMock, dismiss: vi.fn(), toasts: [] }),
}));

vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureException: captureExceptionMock,
  getTraceData: vi.fn(() => ({})),
  withServerActionInstrumentation: vi.fn(
    (_name: string, _options: unknown, callback: () => unknown) => callback(),
  ),
}));

vi.mock("../TTSButton", () => ({
  TTSButton: () => null,
}));

const SESSION_ID = "550e8400-e29b-41d4-a716-446655440000";
const SAVED_MESSAGE_ID = `${SESSION_ID}-seq-7`;
const FEEDBACK_URL = `/api/proxy/api/chat/sessions/${SESSION_ID}/feedback`;

function reply(
  id = SAVED_MESSAGE_ID,
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id,
    role: "assistant",
    parts: [{ type: "text", text: "Here is the summary.", state: "done" }],
  };
}

function captureRequests() {
  const bodies: MessageFeedbackRequest[] = [];
  server.use(
    getPostV2SubmitMessageFeedbackMockHandler200(async ({ request }) => {
      bodies.push((await request.json()) as MessageFeedbackRequest);
      return { id: "feedback-1", langfuse_target: "trace" };
    }),
  );
  return bodies;
}

function failRequests() {
  server.use(
    http.post(FEEDBACK_URL, () =>
      HttpResponse.json({ detail: "Internal Server Error" }, { status: 500 }),
    ),
  );
}

function toastTitles() {
  return toastMock.mock.calls.map(
    ([args]) => (args as { title: string }).title,
  );
}

const upvoteButton = () =>
  screen.getByRole("button", { name: "Good response" });
const downvoteButton = () =>
  screen.getByRole("button", { name: "Bad response" });

beforeEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: vi.fn(async (_text: string) => {}) },
  });
});

afterEach(() => {
  cleanup();
  toastMock.mockReset();
  captureExceptionMock.mockReset();
});

describe("AssistantMessageActions feedback", () => {
  test("thanks the user for a thumbs up only once the rating is saved", async () => {
    let releaseResponse: () => void = () => {};
    const responseHeld = new Promise<void>((resolve) => {
      releaseResponse = resolve;
    });
    const bodies: MessageFeedbackRequest[] = [];
    server.use(
      getPostV2SubmitMessageFeedbackMockHandler200(async ({ request }) => {
        bodies.push((await request.json()) as MessageFeedbackRequest);
        await responseHeld;
        return { id: "feedback-1", langfuse_target: "trace" };
      }),
    );
    render(
      <AssistantMessageActions message={reply()} sessionID={SESSION_ID} />,
    );

    fireEvent.click(upvoteButton());

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({
      message_id: SAVED_MESSAGE_ID,
      score_name: "user-feedback",
      score_value: 1,
    });
    expect(toastTitles()).not.toContain("Thank you for your feedback!");

    releaseResponse();

    await waitFor(() =>
      expect(toastTitles()).toContain("Thank you for your feedback!"),
    );
    expect((downvoteButton() as HTMLButtonElement).disabled).toBe(true);
  });

  test("shows an error and resets the buttons when the rating is not saved", async () => {
    failRequests();
    render(
      <AssistantMessageActions message={reply()} sessionID={SESSION_ID} />,
    );

    fireEvent.click(upvoteButton());

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't save your feedback",
          variant: "destructive",
        }),
      ),
    );
    expect(toastTitles()).not.toContain("Thank you for your feedback!");
    expect(captureExceptionMock).toHaveBeenCalled();
    await waitFor(() =>
      expect((downvoteButton() as HTMLButtonElement).disabled).toBe(false),
    );

    const bodies = captureRequests();
    fireEvent.click(upvoteButton());
    await waitFor(() => expect(bodies).toHaveLength(1));
  });

  test("sends the thumbs-down comment and thanks the user once saved", async () => {
    const bodies = captureRequests();
    render(
      <AssistantMessageActions message={reply()} sessionID={SESSION_ID} />,
    );

    fireEvent.click(downvoteButton());
    fireEvent.change(
      await screen.findByPlaceholderText(
        "Tell us what went wrong or could be improved...",
      ),
      { target: { value: "It edited the wrong file" } },
    );
    fireEvent.click(screen.getByRole("button", { name: "Submit feedback" }));

    await waitFor(() =>
      expect(toastTitles()).toContain("Thank you for your feedback!"),
    );
    expect(bodies).toEqual([
      {
        message_id: SAVED_MESSAGE_ID,
        score_name: "user-feedback",
        score_value: 0,
        comment: "It edited the wrong file",
      },
    ]);
  });

  test("a reply still keyed by the stream's id says why it can't be rated yet", async () => {
    const bodies = captureRequests();
    render(
      <AssistantMessageActions
        message={reply("b7e3c1d2-streamed-reply")}
        sessionID={SESSION_ID}
      />,
    );

    // Clickable, so the click is answered instead of silently swallowed.
    expect((upvoteButton() as HTMLButtonElement).disabled).toBe(false);
    expect(upvoteButton().getAttribute("aria-disabled")).toBe("true");
    expect(downvoteButton().getAttribute("aria-disabled")).toBe("true");

    fireEvent.click(upvoteButton());
    fireEvent.click(downvoteButton());

    expect(toastTitles()).toEqual([
      "Can't rate this reply yet",
      "Can't rate this reply yet",
    ]);
    expect(
      screen.queryByPlaceholderText(
        "Tell us what went wrong or could be improved...",
      ),
    ).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Copy" }));

    await waitFor(() => expect(toastTitles()).toContain("Copied!"));
    expect(bodies).toEqual([]);
  });

  test("a live reply is rated by the saved id the chat learned for it", async () => {
    const bodies = captureRequests();
    render(
      <AssistantMessageActions
        message={reply("turn:turn-1")}
        sessionID={SESSION_ID}
        savedMessageId={`${SESSION_ID}-seq-12`}
      />,
    );

    expect(upvoteButton().getAttribute("aria-disabled")).toBe("false");
    fireEvent.click(upvoteButton());

    await waitFor(() =>
      expect(toastTitles()).toContain("Thank you for your feedback!"),
    );
    expect(bodies).toEqual([
      {
        message_id: `${SESSION_ID}-seq-12`,
        score_name: "user-feedback",
        score_value: 1,
      },
    ]);
  });

  test("a saved id passed in wins over the bubble's own id", async () => {
    const bodies = captureRequests();
    render(
      <AssistantMessageActions
        message={reply()}
        sessionID={SESSION_ID}
        savedMessageId={`${SESSION_ID}-seq-9`}
      />,
    );

    fireEvent.click(upvoteButton());

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0].message_id).toBe(`${SESSION_ID}-seq-9`);
  });

  test("copying a saved reply records a copy without thanking the user", async () => {
    const bodies = captureRequests();
    render(
      <AssistantMessageActions message={reply()} sessionID={SESSION_ID} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Copy" }));

    await waitFor(() => expect(bodies).toHaveLength(1));
    expect(bodies[0]).toEqual({
      message_id: SAVED_MESSAGE_ID,
      score_name: "copy",
      score_value: 1,
    });
    expect(toastTitles()).toEqual(["Copied!"]);
  });
});
