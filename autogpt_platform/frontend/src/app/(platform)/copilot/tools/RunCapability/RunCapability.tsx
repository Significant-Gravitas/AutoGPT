"use client";

import type { ToolUIPart } from "ai";
import { GenericTool } from "../GenericTool/GenericTool";
import { RunBlockTool } from "../RunBlock/RunBlock";
import { RunMCPToolComponent } from "../RunMCPTool/RunMCPTool";
import {
  asBlockPart,
  asMcpPart,
  capabilityId,
  isCapabilityDetails,
  isMcpCapability,
  parseOutputObject,
} from "./helpers";

interface Props {
  part: ToolUIPart;
}

/** run_capability / resume_capability / describe_capability render through
 *  the block or MCP renderer their target belongs to, chosen from the
 *  capability id and the response shape. */
export function RunCapabilityTool({ part }: Props) {
  const output = parseOutputObject(part.output);
  const id = capabilityId(part.input);
  if (isCapabilityDetails(output) || id.startsWith("tool:")) {
    return <GenericTool part={part} />;
  }
  if (isMcpCapability(id, output)) {
    return <RunMCPToolComponent part={asMcpPart(part, output)} />;
  }
  return <RunBlockTool part={asBlockPart(part)} />;
}
