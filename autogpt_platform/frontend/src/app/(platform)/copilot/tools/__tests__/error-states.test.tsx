import type { ToolUIPart } from "ai";
import { describe, expect, it } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { AskQuestionTool } from "../AskQuestion/AskQuestion";
import {
  CreateFeatureRequestTool,
  SearchFeatureRequestsTool,
} from "../FeatureRequests/FeatureRequests";
import { FindAgentsTool } from "../FindAgents/FindAgents";
import { FindBlocksTool } from "../FindBlocks/FindBlocks";
import { getAccordionMeta, type RunBlockToolOutput } from "../RunBlock/helpers";
import { SearchDocsTool } from "../SearchDocs/SearchDocs";
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

// Each failed tool call shows its status line and icon in the error colour.
const failedTools = [
  {
    name: "AskQuestion",
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
    view: (
      <SearchFeatureRequestsTool
        part={failedPart("tool-search_feature_requests")}
      />
    ),
  },
  {
    name: "CreateFeatureRequest",
    view: (
      <CreateFeatureRequestTool
        part={failedPart("tool-create_feature_request")}
      />
    ),
  },
  {
    name: "FindAgents",
    view: <FindAgentsTool part={failedPart("tool-find_agent")} />,
  },
  {
    name: "FindBlocks",
    view: <FindBlocksTool part={failedPart("tool-find_block")} />,
  },
  {
    name: "SearchDocs",
    view: <SearchDocsTool part={failedPart("tool-search_docs")} />,
  },
  {
    name: "ViewAgentOutput",
    view: <ViewAgentOutputTool part={failedPart("tool-view_agent_output")} />,
  },
];

describe("failed tool calls", () => {
  it.each(failedTools)(
    "$name shows its status in the error colour",
    ({ view }) => {
      const { container } = render(view);
      const errorColoured = container.querySelectorAll(".text-error");
      expect(errorColoured.length).toBeGreaterThan(0);
      expect(container.querySelector("svg.text-error")).not.toBeNull();
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
