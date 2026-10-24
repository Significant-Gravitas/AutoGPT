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

  it("waits for the stream to settle before the one-click hand-off", async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <CopilotChatActionsProvider onSend={onSend}>
        <ExpertChangeGroup parts={[hire("a")]} isCurrentlyStreaming />
      </CopilotChatActionsProvider>,
    );
    const name = `Hire James and hand off ${TITLE}`;

    for (const label of [name, "Hire only", "Decline"]) {
      expect(screen.getByRole("button", { name: label })).toHaveProperty(
        "disabled",
        true,
      );
    }
    await user.click(screen.getByRole("button", { name }));
    expect(onSend).not.toHaveBeenCalled();

    rerender(
      <CopilotChatActionsProvider onSend={onSend}>
        <ExpertChangeGroup parts={[hire("a")]} isCurrentlyStreaming={false} />
      </CopilotChatActionsProvider>,
    );
    await user.click(screen.getByRole("button", { name }));
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
            status: "sent",
            sub_session_id: "thread-1",
            link: "/copilot?sessionId=thread-1",
          },
        }}
      />,
    );

    expect(screen.getByText("Expert hired")).toBeDefined();
    expect(
      screen.getByText(/Sent the Printify order pipeline to/),
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

    expect(
      screen.getByText(/may have started in James's chat\. Open it to check\./),
    ).toBeDefined();
    expect(screen.queryByText(/started on/)).toBeNull();
    expect(
      screen
        .getByRole("link", { name: "Open James's chat" })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=thread-1");
  });

  it("never calls a sent task failed just because it has no link", () => {
    render(
      <ExpertChangeCard
        output={{
          type: "expert_change_applied",
          applied: true,
          kind: "hire",
          expert,
          handoff: { title: TITLE, status: "sent", link: null },
        }}
      />,
    );

    expect(
      screen.getByText(/Sent the Printify order pipeline to/),
    ).toBeDefined();
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.queryByRole("link")).toBeNull();
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

describe("hand-off parts follow the theme", () => {
  // Raw palette classes don't follow the dark theme; physical margins don't
  // flip in RTL. Every hand-off state must use tokens and logical sides.
  const RAW =
    /\b(?:bg|text|border|ring|divide)-(?:zinc|neutral|gray|white|black|amber|slate)\b|\bml-|\bmr-/;
  const expert = { id: "exp-james", name: "James", role: "Operations" };

  // The Button atom's own classes belong to the design system, not here.
  function classesIn(container: HTMLElement) {
    return [container, ...container.querySelectorAll<HTMLElement>("[class]")]
      .filter((el) => el.tagName !== "BUTTON")
      .flatMap((el) => el.getAttribute("class")?.split(/\s+/) ?? [])
      .filter((cls) => RAW.test(cls.replace(/^[a-z-]+:/, "")));
  }

  it("uses only design tokens on the proposal card", () => {
    const part = hire("a");
    (part.output as { handoff: { brief: string } }).handoff.brief =
      `${BRIEF} ${BRIEF}`;
    const { container } = render(
      <CopilotChatActionsProvider onSend={onSend}>
        <ExpertChangeGroup parts={[part]} />
      </CopilotChatActionsProvider>,
    );
    const section = screen.getByRole("region", { name: "First task" });
    expect(classesIn(section)).toEqual([]);
    const actions = screen
      .getByRole("button", { name: "Hire only" })
      .closest("div.flex-wrap-reverse") as HTMLElement;
    expect(classesIn(actions)).toEqual([]);
    expect(container).toBeDefined();
  });

  it.each(["sent", "queued", "unconfirmed", "failed"])(
    "uses only design tokens on the %s row",
    (status) => {
      const { container } = render(
        <ExpertChangeCard
          output={{
            type: "expert_change_applied",
            applied: true,
            kind: "hire",
            expert,
            handoff: {
              title: TITLE,
              status,
              link: status === "failed" ? null : "/copilot?sessionId=t",
            },
          }}
        />,
      );
      const row = (screen.queryByRole("status") ??
        screen
          .getByText(/Printify order pipeline/)
          .closest("div")) as HTMLElement;
      expect(classesIn(row)).toEqual([]);
      expect(container).toBeDefined();
    },
  );
});
