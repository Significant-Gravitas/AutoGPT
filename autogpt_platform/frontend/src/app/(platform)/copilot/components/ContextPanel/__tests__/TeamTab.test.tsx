import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../../../copilotStreamStore";
import { useCopilotUIStore } from "../../../store";
import { TeamTab } from "../components/TeamTab/TeamTab";
import { TeamToggle } from "../components/TeamTab/components/TeamToggle";
import { ContextPanel } from "../ContextPanel";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

const SESSION = "session-1";

function toolPart(
  toolName: string,
  toolCallId: string,
  input: unknown,
  output?: unknown,
) {
  return {
    type: `tool-${toolName}`,
    toolCallId,
    state: output === undefined ? "input-available" : "output-available",
    input,
    output,
  } as UIMessage["parts"][number];
}

function setMessages(parts: UIMessage["parts"]) {
  useCopilotStreamStore
    .getState()
    .setMessageSnapshot(SESSION, [{ id: "a", role: "assistant", parts }]);
}

function idleSubSession(id: string) {
  return getGetV2GetSessionMockHandler200({
    id,
    created_at: "2026-08-21T00:00:00Z",
    updated_at: "2026-08-21T00:00:00Z",
    user_id: "u-1",
    chat_status: "idle",
    messages: [],
  });
}

beforeEach(() => {
  useCopilotStreamStore.getState().resetAll();
  useCopilotUIStore.setState((s) => ({
    artifactPanel: {
      ...s.artifactPanel,
      isOpen: false,
      activeArtifact: null,
      activeTab: "files",
      history: [],
      lastArtifact: null,
    },
  }));
});

afterEach(() => {
  useCopilotStreamStore.getState().resetAll();
});

describe("TeamTab", () => {
  it("shows the empty state when the chat delegated to nobody", () => {
    render(<TeamTab sessionId={SESSION} />);
    expect(screen.getByText("No teammates at work")).toBeDefined();
  });

  it("lists one row per delegated teammate with the expert's name, role and brief", () => {
    setMessages([
      toolPart(
        "delegate_to_expert",
        "c1",
        { expert_id: "exp-1", prompt: "Find three vendors" },
        {
          status: "completed",
          sub_session_id: "sub-1",
          response: "Vendor list attached",
          elapsed_seconds: 65,
          expert: { id: "exp-1", name: "Mira", role: "Researcher" },
        },
      ),
      toolPart(
        "handoff_to_expert",
        "c2",
        { expert_id: "exp-2", prompt: "Own the launch email" },
        {
          status: "transferred",
          sub_session_id: "sub-2",
          expert: { id: "exp-2", name: "Theo", role: "Writer" },
        },
      ),
    ]);
    render(<TeamTab sessionId={SESSION} withHeader />);

    const rows = screen.getAllByTestId("delegation-row");
    expect(rows).toHaveLength(2);
    expect(rows[0].textContent).toContain("Mira");
    expect(rows[0].textContent).toContain("Researcher");
    expect(rows[0].textContent).toContain("Vendor list attached");
    expect(rows[0].textContent).toContain("Completed");
    expect(rows[0].textContent).toContain("1m 05s");
    expect(rows[1].textContent).toContain("Theo");
    expect(rows[1].textContent).toContain("Handed over");
    expect(rows[1].textContent).toContain("Own the launch email");
    expect(
      screen.getByLabelText("Open Theo's session").getAttribute("href"),
    ).toBe("/copilot?sessionId=sub-2");
    expect(screen.getByText("2 settled")).toBeDefined();
  });

  it("counts a still-running delegate call as working", () => {
    setMessages([
      toolPart("delegate_to_expert", "c1", {
        expert_id: "exp-1",
        prompt: "Draft the brief",
      }),
    ]);
    render(<TeamTab sessionId={SESSION} withHeader />);

    expect(screen.getByText("1 working")).toBeDefined();
    expect(screen.getByTestId("delegation-row").textContent).toContain(
      "Draft the brief",
    );
  });

  it("flips a frozen running status to completed once the polled sub-session goes idle", async () => {
    server.use(idleSubSession("sub-1"));
    setMessages([
      toolPart(
        "delegate_to_expert",
        "c1",
        { expert_id: "exp-1", prompt: "Draft" },
        { status: "running", sub_session_id: "sub-1" },
      ),
    ]);
    render(<TeamTab sessionId={SESSION} />);

    await waitFor(() =>
      expect(screen.getByTestId("delegation-row").textContent).toContain(
        "Completed",
      ),
    );
    expect(screen.queryByText(/working/)).toBeNull();
  });
});

describe("TeamToggle", () => {
  it("renders nothing until the chat delegates to someone", () => {
    render(<TeamToggle sessionId={SESSION} />);
    expect(screen.queryByLabelText(/team/i)).toBeNull();
  });

  it("badges the live count and opens the team tab", () => {
    setMessages([
      toolPart("delegate_to_expert", "c1", { expert_id: "exp-1" }),
      toolPart(
        "delegate_to_expert",
        "c2",
        { expert_id: "exp-2" },
        { status: "completed", sub_session_id: "sub-2" },
      ),
    ]);
    render(<TeamToggle sessionId={SESSION} />);

    const button = screen.getByLabelText("Open team, 1 working");
    expect(button.textContent).toContain("1");
    fireEvent.click(button);

    const panel = useCopilotUIStore.getState().artifactPanel;
    expect(panel.isOpen).toBe(true);
    expect(panel.activeTab).toBe("team");
  });

  it("drops the badge while the roster is on screen", () => {
    setMessages([toolPart("delegate_to_expert", "c1", { expert_id: "exp-1" })]);
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, isOpen: true, activeTab: "team" },
    }));
    render(<TeamToggle sessionId={SESSION} />);

    const button = screen.getByLabelText("Hide team");
    expect(button.getAttribute("aria-pressed")).toBe("true");
    expect(button.textContent).toBe("");
  });
});

describe("ContextPanel team tab", () => {
  it("docks for the team tab and renders the roster", async () => {
    setMessages([toolPart("delegate_to_expert", "c1", { expert_id: "exp-1" })]);
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, isOpen: true, activeTab: "team" },
    }));
    const { container } = render(<ContextPanel sessionId={SESSION} />);

    await waitFor(() =>
      expect(container.querySelector("[data-context-panel]")).not.toBeNull(),
    );
    expect(screen.getByText("Team")).toBeDefined();
    expect(screen.getByTestId("delegation-row")).toBeDefined();
  });

  it("mobile: opens the sheet titled Team", async () => {
    useCopilotUIStore.setState((s) => ({
      artifactPanel: { ...s.artifactPanel, isOpen: true, activeTab: "team" },
    }));
    render(<ContextPanel sessionId={SESSION} mobile />);

    expect(await screen.findByRole("dialog")).toBeDefined();
    expect(screen.getByRole("heading", { name: "Team" })).toBeDefined();
  });
});
