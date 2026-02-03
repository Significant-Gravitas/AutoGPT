import { ToolUIPart } from "ai";
import {
  FindBlockInput,
  FindBlockOutput,
  FindBlockToolPart,
} from "./FindBlocks";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

export const getAnimationText = (part: FindBlockToolPart): string => {
  switch (part.state) {
    case "input-streaming":
      return "Searching blocks for you";

    case "input-available": {
      const query = (part.input as FindBlockInput).query;
      return `Finding "${query}" blocks`;
    }

    case "output-available": {
      const parsed = JSON.parse(part.output as string) as FindBlockOutput;
      if (parsed) {
        return `Found ${parsed.count} "${(part.input as FindBlockInput).query}" blocks`;
      }
      return "Found blocks";
    }

    case "output-error":
      return "Error finding blocks";

    default:
      return "Processing";
  }
};

export const StateIcon = ({ state }: { state: ToolUIPart["state"] }) => {
  switch (state) {
    case "input-streaming":
    case "input-available":
      return (
        <CircleNotchIcon
          className="h-4 w-4 animate-spin text-muted-foreground"
          weight="bold"
        />
      );
    case "output-available":
      return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
    case "output-error":
      return <XCircleIcon className="h-4 w-4 text-red-500" />;
    default:
      return null;
  }
};
