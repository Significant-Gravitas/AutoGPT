import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import type { ToolUIPart } from "ai";
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
      preview: { kind: "hire", name, role: "Operations" },
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

describe("hiring an expert for a task already in the chat", () => {
  beforeEach(() => onSend.mockClear());

  it("shows the task and offers hire and hand off as the main answer", () => {
    renderGroup([hire("a")]);

    expect(screen.getByRole("region", { name: "First task" })).toBeDefined();
    expect(screen.getByText(TITLE)).toBeDefined();
    expect(
      screen.getByRole("button", {
        name: `Hire James and hand off ${TITLE}`,
      }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Hire only" })).toBeDefined();
    expect(screen.getByRole("button", { name: "Decline" })).toBeDefined();
    expect(screen.queryByRole("button", { name: "Approve" })).toBeNull();
  });

  it("hires and hands off in one click", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a")]);

    await user.click(
      screen.getByRole("button", { name: `Hire James and hand off ${TITLE}` }),
    );

    // The hand-off id: confirm_expert_change hires AND starts the task.
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith(
      `Approved: hire James to take on ${TITLE} (confirmation_id: h-a).`,
    );
    expect(screen.getByText("Sent")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Hire only" })).toBeNull();
  });

  it("hires without the task on Hire only", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a")]);

    await user.click(screen.getByRole("button", { name: "Hire only" }));

    expect(onSend).toHaveBeenCalledWith(
      "Approved: hire James (confirmation_id: c-a).",
    );
  });

  it("declines the hire, not just the task", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a")]);

    await user.click(screen.getByRole("button", { name: "Decline" }));

    expect(onSend).toHaveBeenCalledWith(
      "Not approved: do not hire James, discard that proposal (confirmation_id: c-a).",
    );
  });

  it("keeps each expert's choice when several are sent together", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a"), hire("b", "Nadia")]);

    await user.click(
      screen.getByRole("button", { name: `Hire James and hand off ${TITLE}` }),
    );
    await user.click(screen.getByRole("button", { name: "Hire only" }));
    expect(onSend).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Send decisions" }));

    expect(onSend).toHaveBeenCalledWith(
      `Approved: hire James to take on ${TITLE} (confirmation_id: h-a).\n` +
        "Approved: hire Nadia (confirmation_id: c-b).",
    );
  });

  it("keeps the plain Approve card for a hire with no task", async () => {
    const user = userEvent.setup();
    renderGroup([hire("a", "James", false)]);

    expect(screen.queryByRole("region", { name: "First task" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Hire only" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Approve" }));

    expect(onSend).toHaveBeenCalledWith(
      "Approved: hire James (confirmation_id: c-a).",
    );
  });

  it("lets the whole task be read when it is long", async () => {
    const user = userEvent.setup();
    const long = `${BRIEF} ${BRIEF}`;
    const part = hire("a");
    (part.output as { handoff: { brief: string } }).handoff.brief = long;
    renderGroup([part]);

    const toggle = screen.getByRole("button", { name: "Show all" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    await user.click(toggle);
    expect(
      screen
        .getByRole("button", { name: "Show less" })
        .getAttribute("aria-expanded"),
    ).toBe("true");
  });
});

describe("after the hire and hand-off", () => {
  const expert = { id: "exp-james", name: "James", role: "Operations" };

  it("says where the work is happening and links to the expert's chat", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_applied",
          applied: true,
          kind: "hire",
          expert,
          handoff: {
            title: TITLE,
            status: "started",
            sub_session_id: "thread-1",
            link: "/copilot?sessionId=thread-1",
          },
        }}
      />,
    );

    expect(screen.getByText("Expert hired")).toBeDefined();
    expect(
      screen.getByText(/started on the Printify order pipeline/),
    ).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Open James's chat" })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=thread-1");
  });

  it("says the task waits for a slot rather than that it started", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_applied",
          applied: true,
          kind: "hire",
          expert,
          handoff: {
            title: TITLE,
            status: "queued",
            sub_session_id: "thread-1",
            link: "/copilot?sessionId=thread-1",
          },
        }}
      />,
    );

    expect(
      screen.getByText(/will start on the Printify order pipeline/),
    ).toBeDefined();
    expect(screen.queryByText(/started on/)).toBeNull();
    expect(
      screen
        .getByRole("link", { name: "Open James's chat" })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=thread-1");
  });

  it("points at the thread when it is not known whether the task started", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_applied",
          applied: true,
          kind: "hire",
          expert,
          handoff: {
            title: TITLE,
            status: "unconfirmed",
            sub_session_id: "thread-1",
            link: "/copilot?sessionId=thread-1",
          },
        }}
      />,
    );

    expect(screen.getByText(/was sent to James's chat/)).toBeDefined();
    expect(screen.queryByText(/started on/)).toBeNull();
    expect(
      screen
        .getByRole("link", { name: "Open James's chat" })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=thread-1");
  });

  it("says plainly when the task did not start", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_applied",
          applied: true,
          kind: "hire",
          expert,
          handoff: { title: TITLE, status: "failed", link: null },
        }}
      />,
    );

    expect(screen.getByRole("status").textContent).toBe(
      `James is hired, but ${TITLE} didn't start.`,
    );
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("shows nothing extra for a hire saved without a hand-off", () => {
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

    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.queryByRole("status")).toBeNull();
  });
});
