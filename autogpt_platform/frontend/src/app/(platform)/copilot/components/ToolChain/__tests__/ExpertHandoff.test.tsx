import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart } from "ai";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ExpertChangeCard, ExpertChangeGroup } from "../ExpertCards";

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: vi.fn(),
  useToast: () => ({ toast: vi.fn(), dismiss: vi.fn() }),
}));

const onSend = vi.fn();
const TITLE = "the Printify order pipeline";
const BRIEF =
  "Own the Printify order pipeline: move new Etsy orders into Printify and sync tracking numbers back to Etsy every morning.";
const TAGLINE = "Runs your operating rhythm, SOPs and vendors.";
const HAND_OFF = "Hire and start";

function hire(id: string, name = "James", withTask = true): ToolUIPart {
  return {
    type: "tool-hire_expert",
    state: "output-available",
    toolCallId: `call-${id}`,
    input: {},
    output: {
      type: "expert_change_proposed",
      applied: false,
      confirmation_id: `c-${id}`,
      ...(withTask
        ? {
            handoff: { title: TITLE, brief: BRIEF },
            handoff_confirmation_id: `h-${id}`,
          }
        : {}),
      preview: { kind: "hire", name, role: "Operations", tagline: TAGLINE },
    },
  } as ToolUIPart;
}

function renderGroup(parts: ToolUIPart[]) {
  return render(
    <CopilotChatActionsProvider onSend={onSend}>
      <ExpertChangeGroup parts={parts} />
    </CopilotChatActionsProvider>,
  );
}

function isHidden(text: string) {
  return !!screen.getByText(text, { exact: false }).closest("[hidden]");
}

describe("hiring an expert for a task already in the chat", () => {
  beforeEach(() => onSend.mockClear());

  it("asks one question and offers hire and start as the main answer", () => {
    renderGroup([hire("a")]);

    expect(screen.getByText(`Hire James to take on ${TITLE}?`)).toBeDefined();
    expect(screen.getByRole("button", { name: HAND_OFF })).toBeDefined();
    expect(screen.getByRole("button", { name: "Hire only" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Decline" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  });

  it("keeps the brief and the charter behind one disclosure", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a")]);

    const toggle = screen.getByRole("button", { name: "What James will get" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(isHidden(BRIEF)).toBe(true);
    expect(isHidden(TAGLINE)).toBe(true);

    await user.click(toggle);

    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(isHidden(BRIEF)).toBe(false);
    expect(isHidden(TAGLINE)).toBe(false);
  });

  it("hires and hands off in one click, then shrinks to a receipt", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a")]);

    await user.click(screen.getByRole("button", { name: HAND_OFF }));

    // The hand-off id: confirm_expert_change hires AND starts the task.
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith(
      `Approved: hire James to take on ${TITLE} (confirmation_id: h-a).`,
    );
    expect(
      screen.getByText(`You hired James and handed off ${TITLE}.`),
    ).toBeDefined();
    expect(screen.queryByText(`Hire James to take on ${TITLE}?`)).toBeNull();
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("waits for the stream to settle before the one-click hand-off", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <CopilotChatActionsProvider onSend={onSend}>
        <ExpertChangeGroup parts={[hire("a")]} isCurrentlyStreaming />
      </CopilotChatActionsProvider>,
    );

    for (const label of [HAND_OFF, "Hire only", "Decline"]) {
      expect(screen.getByRole("button", { name: label })).toHaveProperty(
        "disabled",
        true,
      );
    }
    await user.click(screen.getByRole("button", { name: HAND_OFF }));
    expect(onSend).not.toHaveBeenCalled();

    rerender(
      <CopilotChatActionsProvider onSend={onSend}>
        <ExpertChangeGroup parts={[hire("a")]} isCurrentlyStreaming={false} />
      </CopilotChatActionsProvider>,
    );
    await user.click(screen.getByRole("button", { name: HAND_OFF }));
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith(
      `Approved: hire James to take on ${TITLE} (confirmation_id: h-a).`,
    );
  });

  it("hires without the task on Hire only", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a")]);

    await user.click(screen.getByRole("button", { name: "Hire only" }));

    expect(onSend).toHaveBeenCalledWith(
      "Approved: hire James (confirmation_id: c-a).",
    );
    expect(screen.getByText("You hired James.")).toBeDefined();
  });

  it("declines the hire, not just the task", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a")]);

    await user.click(screen.getByRole("button", { name: "Decline" }));

    expect(onSend).toHaveBeenCalledWith(
      "Not approved: do not hire James, discard that proposal (confirmation_id: c-a).",
    );
    expect(screen.getByText("You declined hiring James.")).toBeDefined();
  });

  it("keeps each expert's choice when several are sent together", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a"), hire("b", "Nadia")]);

    await user.click(screen.getByRole("button", { name: HAND_OFF }));
    await user.click(screen.getByRole("button", { name: "Hire only" }));
    expect(onSend).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    expect(onSend).toHaveBeenCalledWith(
      `Approved: hire James to take on ${TITLE} (confirmation_id: h-a).\n` +
        "Approved: hire Nadia (confirmation_id: c-b).",
    );
    // Several experts keep their pager after the send, not one receipt.
    expect(screen.getByText("Sent")).toBeDefined();
  });

  it("shows which choice was made when paging back before the send", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a"), hire("b", "Nadia")]);

    await user.click(screen.getByRole("button", { name: HAND_OFF }));
    await user.click(screen.getByRole("button", { name: "Hire only" }));
    expect(screen.getByText("Hire only chosen")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Previous expert" }));
    expect(screen.getByText("Hire and hand off chosen")).toBeDefined();
    expect(screen.getByRole("button", { name: "Unapprove" })).toBeDefined();
    expect(screen.queryByRole("button", { name: HAND_OFF })).toBeNull();
    expect(onSend).not.toHaveBeenCalled();
  });

  it("keeps the plain Approve card for a hire with no task", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a", "James", false)]);

    expect(screen.queryByText(/What James will get/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Hire only" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Approve" }));

    expect(onSend).toHaveBeenCalledWith(
      "Approved: hire James (confirmation_id: c-a).",
    );
    expect(screen.queryByText(/You hired James/)).toBeNull();
  });
});

function thread(chatStatus: string) {
  return getGetV2GetSessionMockHandler200({
    id: "thread-1",
    created_at: "2026-10-26T00:00:00Z",
    updated_at: "2026-10-26T00:00:00Z",
    user_id: "u-1",
    messages: [],
    chat_status: chatStatus,
    active_stream: null,
  });
}

const expert = { id: "exp-james", name: "James", role: "Operations" };

function appliedOutput(
  status: string,
  link: string | null = "/copilot?sessionId=thread-1",
) {
  return {
    type: "expert_change_applied",
    applied: true,
    kind: "hire",
    expert,
    handoff: {
      title: TITLE,
      status,
      ...(link ? { sub_session_id: "thread-1" } : {}),
      link,
    },
  };
}

function sentCard(status: "sent" | "queued" | "unconfirmed") {
  return <ExpertChangeCard output={appliedOutput(status)} />;
}

function chatLink() {
  return screen
    .getByRole("link", { name: "Open James's chat" })
    .getAttribute("href");
}

describe("after the hire and hand-off", () => {
  beforeEach(() => onSend.mockClear());

  it("says who is working on what and links to their chat, nothing more", async () => {
    server.use(thread("running"));
    render(sentCard("sent"));

    const line = await screen.findByText(/is working on/);
    expect(line.textContent).toBe(`James is working on ${TITLE}.`);
    expect(chatLink()).toBe("/copilot?sessionId=thread-1");
    // The proposal above already records the hire.
    expect(screen.queryByText("Expert hired")).toBeNull();
    expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
  });

  it("says the task waits for a slot rather than that it started", async () => {
    server.use(thread("queued"));
    render(sentCard("queued"));

    expect(
      screen.getByText(
        /starts on the Printify order pipeline as soon as one of your other tasks finishes/,
      ),
    ).toBeDefined();
    expect(screen.queryByText(/is working on/)).toBeNull();
    expect(chatLink()).toBe("/copilot?sessionId=thread-1");
  });

  it("points at the thread when it is not known whether the task started", () => {
    render(sentCard("unconfirmed"));

    expect(
      screen.getByText(/may have started in James's chat\. Open it to check\./),
    ).toBeDefined();
    expect(screen.queryByText(/is working on/)).toBeNull();
    expect(chatLink()).toBe("/copilot?sessionId=thread-1");
  });

  it("shows a queued task start working once the expert picks it up", async () => {
    // Mirrors POLL_MS in SubSessionLive.tsx: the thread is re-read this often.
    const POLL_MS = 3000;
    let chatStatus = "queued";
    server.use(
      http.get("/api/proxy/api/chat/sessions/:sessionId", () =>
        HttpResponse.json({
          id: "thread-1",
          created_at: "2026-10-26T00:00:00Z",
          updated_at: "2026-10-26T00:00:00Z",
          user_id: "u-1",
          messages: [],
          chat_status: chatStatus,
          active_stream: null,
        }),
      ),
    );
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      render(sentCard("queued"));

      expect(await screen.findByText(/starts on/)).toBeDefined();

      chatStatus = "running";
      await vi.advanceTimersByTimeAsync(POLL_MS + 100);

      expect(await screen.findByText(/is working on/)).toBeDefined();
      expect(screen.queryByText(/starts on/)).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it("follows the expert's thread and says when they stop working", async () => {
    server.use(thread("idle"));
    render(sentCard("sent"));

    const line = await screen.findByText(/stopped working on/);
    expect(line.textContent).toBe(
      `James stopped working on ${TITLE}. Their chat has the result, or a question for you.`,
    );
    expect(screen.queryByText(/is working on/)).toBeNull();
    expect(chatLink()).toBe("/copilot?sessionId=thread-1");
  });

  it("does not claim the task is running when the thread can't be read", async () => {
    server.use(
      http.get(
        "/api/proxy/api/chat/sessions/:sessionId",
        () => new HttpResponse(null, { status: 500 }),
      ),
    );
    render(sentCard("sent"));

    expect(await screen.findByText("Status unknown")).toBeDefined();
    expect(screen.queryByTestId("handoff-badge")).toBeNull();
  });

  it("never reads an idle thread as a reply when the start is unconfirmed", async () => {
    let polled = false;
    server.use(
      http.get("/api/proxy/api/chat/sessions/:sessionId", () => {
        polled = true;
        return HttpResponse.json({});
      }),
    );
    render(sentCard("unconfirmed"));

    expect(screen.getByText(/may have started in James's chat/)).toBeDefined();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(polled).toBe(false);
    expect(screen.queryByText(/stopped working on/)).toBeNull();
  });

  it("never calls a sent task failed just because it has no link", () => {
    render(<ExpertChangeCard output={appliedOutput("sent", null)} />);

    expect(screen.getByText(/is working on/)).toBeDefined();
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("says plainly when the task did not start", () => {
    render(<ExpertChangeCard output={appliedOutput("failed", null)} />);

    expect(screen.getByRole("status").textContent).toBe(
      `James is hired, but ${TITLE} didn't start.`,
    );
    expect(screen.queryByRole("link")).toBeNull();
    // Read-only: nothing to send from.
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });

  it("offers to try again when the task did not start", async () => {
    const user = userEvent.setup();
    renderGroup([
      {
        type: "tool-confirm_expert_change",
        state: "output-available",
        toolCallId: "call-confirm",
        input: {},
        output: appliedOutput("failed", null),
      } as ToolUIPart,
    ]);

    await user.click(screen.getByRole("button", { name: "Try again" }));

    expect(onSend).toHaveBeenCalledWith(`Try handing ${TITLE} to James again.`);
    expect(screen.getByText("Asked Otto to try again.")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });

  it("keeps the full card for a hire saved without a hand-off", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_applied",
          applied: true,
          kind: "hire",
          expert,
        }}
      />,
    );

    expect(screen.getByText("Expert hired")).toBeDefined();
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.queryByRole("status")).toBeNull();
  });
});

describe("hand-off parts follow the theme", () => {
  // Raw palette classes don't follow the dark theme; physical margins don't
  // flip in RTL. Every hand-off state must use tokens and logical sides.
  const RAW =
    /\b(?:bg|text|border|ring|divide)-(?:zinc|neutral|gray|white|black|amber|slate)\b|\bml-|\bmr-|\bpl-|\bpr-/;

  // The Button atom's and the avatar's own classes belong to the design
  // system, not here.
  function classesIn(container: HTMLElement) {
    return [container, ...container.querySelectorAll<HTMLElement>("[class]")]
      .filter(
        (el) =>
          el.tagName !== "BUTTON" &&
          !el.closest("[data-slot='expert-avatar'], [data-slot='avatar']"),
      )
      .flatMap((el) => el.getAttribute("class")?.split(/\s+/) ?? [])
      .filter((cls) => RAW.test(cls.replace(/^[a-z-]+:/, "")));
  }

  it("uses only design tokens on the proposal card", () => {
    const { container } = renderGroup([hire("a")]);
    expect(classesIn(container)).toEqual([]);
  });

  it("uses only design tokens on the receipt", async () => {
    const user = userEvent.setup();
    const { container } = renderGroup([hire("a")]);
    await user.click(screen.getByRole("button", { name: HAND_OFF }));
    expect(classesIn(container)).toEqual([]);
  });

  it.each(["sent", "queued", "unconfirmed", "failed"])(
    "uses only design tokens on the %s status",
    (status) => {
      const { container } = render(
        <ExpertChangeCard
          output={appliedOutput(
            status,
            status === "failed" ? null : "/copilot?sessionId=t",
          )}
        />,
      );
      expect(classesIn(container)).toEqual([]);
    },
  );
});
