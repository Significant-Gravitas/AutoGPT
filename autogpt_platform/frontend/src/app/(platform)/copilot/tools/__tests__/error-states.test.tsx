import type { ToolUIPart } from "ai";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { AskQuestionTool } from "../AskQuestion/AskQuestion";
import { ConnectIntegrationTool } from "../ConnectIntegrationTool/ConnectIntegrationTool";
import { CreateAgentTool } from "../CreateAgent/CreateAgent";
import { ToolIcon as DecomposeGoalToolIcon } from "../DecomposeGoal/helpers";
import { EditAgentTool } from "../EditAgent/EditAgent";
import {
  CreateFeatureRequestTool,
  SearchFeatureRequestsTool,
} from "../FeatureRequests/FeatureRequests";
import { FindAgentsTool } from "../FindAgents/FindAgents";
import { FindBlocksTool } from "../FindBlocks/FindBlocks";
import { connectionBadge } from "../FindCapabilities/helpers";
import { FolderTool } from "../FolderTool/FolderTool";
import { GenericTool } from "../GenericTool/GenericTool";
import { RunAgentTool } from "../RunAgent/RunAgent";
import { getAccordionMeta, type RunBlockToolOutput } from "../RunBlock/helpers";
import { RunBlockTool } from "../RunBlock/RunBlock";
import { RunMCPToolComponent } from "../RunMCPTool/RunMCPTool";
import { SearchDocsTool } from "../SearchDocs/SearchDocs";
import { SetupTriggerTool } from "../SetupTrigger/SetupTrigger";
import { ViewAgentOutputTool } from "../ViewAgentOutput/ViewAgentOutput";

function failedPart(type: string) {
  return {
    type,
    toolCallId: `call-${type}`,
    state: "output-error" as const,
    input: {},
    errorText: "Something went wrong",
  };
}

function withChatActions(node: ReactNode) {
  return (
    <CopilotChatActionsProvider onSend={() => {}}>
      {node}
    </CopilotChatActionsProvider>
  );
}

// Each failed tool call shows its status line and icon in the error colour.
const failedTools = [
  {
    name: "AskQuestion",
    status: "Failed to ask question",
    view: (
      <CopilotChatActionsProvider onSend={() => {}}>
        <AskQuestionTool
          part={failedPart("tool-ask_question") as unknown as ToolUIPart}
        />
      </CopilotChatActionsProvider>
    ),
  },
  {
    name: "SearchFeatureRequests",
    status: "Error searching feature requests",
    view: (
      <SearchFeatureRequestsTool
        part={failedPart("tool-search_feature_requests")}
      />
    ),
  },
  {
    name: "CreateFeatureRequest",
    status: "Error creating feature request",
    view: (
      <CreateFeatureRequestTool
        part={failedPart("tool-create_feature_request")}
      />
    ),
  },
  {
    name: "FindAgents",
    status: "Error finding workflows",
    view: <FindAgentsTool part={failedPart("tool-find_agent")} />,
  },
  {
    name: "FindBlocks",
    status: "Search failed",
    view: <FindBlocksTool part={failedPart("tool-find_block")} />,
  },
  {
    name: "SearchDocs",
    status: "Error searching documentation",
    view: <SearchDocsTool part={failedPart("tool-search_docs")} />,
  },
  {
    name: "ViewAgentOutput",
    status: "Error loading workflow output",
    view: <ViewAgentOutputTool part={failedPart("tool-view_agent_output")} />,
  },
  {
    name: "FolderTool",
    status: "Folder operation failed",
    view: (
      <FolderTool
        part={failedPart("tool-list_folders") as unknown as ToolUIPart}
      />
    ),
  },
  {
    name: "GenericTool",
    status: "Command failed",
    view: (
      <GenericTool
        part={failedPart("tool-bash_exec") as unknown as ToolUIPart}
      />
    ),
  },
  {
    name: "RunAgent",
    status: "Something went wrong",
    view: <RunAgentTool part={failedPart("tool-run_agent")} />,
  },
  {
    name: "RunMCPTool",
    status: "Connection error",
    view: <RunMCPToolComponent part={failedPart("tool-run_mcp_tool")} />,
  },
  {
    name: "SetupTrigger",
    status: "Something went wrong",
    view: (
      <SetupTriggerTool part={failedPart("tool-setup_agent_webhook_trigger")} />
    ),
  },
  {
    name: "CreateAgent",
    status: "Error creating workflow",
    view: withChatActions(
      <CreateAgentTool part={failedPart("tool-create_agent")} />,
    ),
  },
  {
    name: "EditAgent",
    status: "Error editing workflow",
    view: withChatActions(
      <EditAgentTool part={failedPart("tool-edit_agent")} />,
    ),
  },
  {
    name: "RunBlock",
    status: "Action failed",
    view: <RunBlockTool part={failedPart("tool-run_block")} />,
  },
  {
    name: "ConnectIntegration",
    status: "Failed to connect Gmail",
    icon: false,
    view: (
      <ConnectIntegrationTool
        part={
          {
            ...failedPart("tool-connect_integration"),
            input: { provider: "Gmail" },
          } as unknown as ToolUIPart
        }
      />
    ),
  },
];

describe("failed tool calls", () => {
  it.each(failedTools)(
    "$name shows its status in the error colour",
    ({ view, status, icon }) => {
      const { container } = render(view);
      expect(screen.getByText(status).classList.contains("text-error")).toBe(
        true,
      );
      if (icon !== false) {
        expect(container.querySelector("svg.text-error")).not.toBeNull();
      }
    },
  );
});

describe("RunBlock accordion for an error output", () => {
  it("titles the accordion Error in the error colour", () => {
    const meta = getAccordionMeta({
      type: "error",
      message: "Block failed",
    } as unknown as RunBlockToolOutput);
    expect(meta.title).toBe("Error");
    expect(meta.titleClassName).toBe("text-error");
    const { container } = render(<>{meta.icon}</>);
    expect(container.querySelector("svg.text-error")).not.toBeNull();
  });
});

describe("FindCapabilities connection badge", () => {
  it("asks the user to sign in when the capability is not connected", () => {
    const badge = connectionBadge({
      id: "gmail",
      name: "Gmail",
      purpose: "Read mail",
      kind: "block",
      connected: false,
    });
    expect(badge?.label).toBe("sign in");
    expect(badge?.className).toBe("bg-surface-hover text-foreground-tertiary");
  });
});

describe("DecomposeGoal tool icon", () => {
  // The card shows ToolErrorCard when the call fails, so check the icon alone.
  it("uses the error colour for a failed call", () => {
    const { container } = render(
      <DecomposeGoalToolIcon isStreaming={false} isError />,
    );
    expect(container.querySelector("svg.text-error")).not.toBeNull();
  });
});
