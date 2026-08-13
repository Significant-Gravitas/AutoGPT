import { render, screen } from "@/tests/integrations/test-utils";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { AgentActivityDropdown } from "../AgentActivityDropdown";
import { AgentExecutionWithInfo } from "../helpers";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseAgentActivityDropdown = vi.hoisted(() => vi.fn());

vi.mock("../useAgentActivityDropdown", () => ({
  useAgentActivityDropdown: mockUseAgentActivityDropdown,
}));

function makeExecution(
  overrides: Partial<AgentExecutionWithInfo> = {},
): AgentExecutionWithInfo {
  return {
    id: "exec-1",
    graph_id: "graph-1",
    status: AgentExecutionStatus.RUNNING,
    started_at: new Date(),
    ended_at: null,
    user_id: "user-1",
    graph_version: 1,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    stats: null,
    agent_name: "Test Agent",
    agent_description: "A running agent",
    library_agent_id: "library-1",
    ...overrides,
  };
}

describe("AgentActivityDropdown", () => {
  beforeEach(() => {
    mockUseAgentActivityDropdown.mockReturnValue({
      activeExecutions: [makeExecution(), makeExecution({ id: "exec-2" })],
      recentCompletions: [],
      recentFailures: [],
      totalCount: 2,
      isReady: true,
      error: null,
      isOpen: false,
      setIsOpen: vi.fn(),
    });
  });

  test("shows the active execution badge count", () => {
    render(<AgentActivityDropdown />);

    expect(screen.getByTestId("agent-activity-badge").textContent).toContain(
      "2",
    );
    expect(screen.getByTestId("agent-activity-button")).toBeDefined();
  });

  test("renders the dropdown content when open", async () => {
    mockUseAgentActivityDropdown.mockReturnValue({
      activeExecutions: [makeExecution()],
      recentCompletions: [],
      recentFailures: [],
      totalCount: 1,
      isReady: true,
      error: null,
      isOpen: true,
      setIsOpen: vi.fn(),
    });

    render(<AgentActivityDropdown />);

    expect(screen.getByTestId("agent-activity-dropdown")).toBeDefined();
    expect(await screen.findByText("Test Agent")).toBeDefined();
  });
});
