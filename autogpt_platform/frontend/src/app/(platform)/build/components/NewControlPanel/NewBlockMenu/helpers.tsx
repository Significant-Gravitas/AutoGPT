import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { BlockUIType } from "../../types";
import { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { BlockCategory } from "../../helper";
import { RJSFSchema } from "@rjsf/utils";
import { SpecialBlockID } from "@/lib/autogpt-server-api";
import { Text } from "@/components/atoms/Text/Text";
import type { Variant } from "@/components/atoms/Text/helpers";

// `variant` is the bold match of the <Text> the copy sits in: body-bold in a
// body-medium title, small-bold-loose in a small-loose description.
export const highlightText = (
  text: string | undefined,
  highlight: string | undefined,
  variant: Variant = "body-bold",
) => {
  if (!text || !highlight) return text;

  function escapeRegExp(s: string) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  const escaped = escapeRegExp(highlight);
  const parts = text.split(new RegExp(`(${escaped})`, "gi"));
  return parts.map((part, i) =>
    part.toLowerCase() === highlight?.toLowerCase() ? (
      <mark key={i} className="bg-transparent">
        <Text variant={variant} as="span" tone="inherit">
          {part}
        </Text>
      </mark>
    ) : (
      part
    ),
  );
};

export const convertLibraryAgentIntoCustomNode = (
  agent: LibraryAgent,
  inputSchema: RJSFSchema = {} as RJSFSchema,
  outputSchema: RJSFSchema = {} as RJSFSchema,
) => {
  const block: BlockInfo = {
    id: SpecialBlockID.AGENT,
    name: agent.name,
    description:
      `Ver.${agent.graph_version}` +
      (agent.description ? ` | ${agent.description}` : ""),
    categories: [{ category: BlockCategory.AGENT, description: "" }],
    inputSchema: inputSchema,
    outputSchema: outputSchema,
    staticOutput: false,
    uiType: BlockUIType.AGENT,
    costs: [],
    contributors: [],
  };

  const hardcodedValues: Record<string, any> = {
    graph_id: agent.graph_id,
    graph_version: agent.graph_version,
    input_schema: inputSchema,
    output_schema: outputSchema,
    agent_name: agent.name,
  };

  return {
    block,
    hardcodedValues,
  };
};
