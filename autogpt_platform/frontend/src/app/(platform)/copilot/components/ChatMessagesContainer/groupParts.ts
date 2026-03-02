import { UIDataTypes, UIMessage, UITools } from "ai";

type Part = UIMessage<unknown, UIDataTypes, UITools>["parts"][number];

export interface SingleItem {
  kind: "single";
  part: Part;
  partIndex: number;
}

export interface GroupItem {
  kind: "group";
  toolType: string;
  parts: Part[];
  partIndices: number[];
}

export type GroupedItem = SingleItem | GroupItem;

/**
 * Groups consecutive message parts of the same tool type into collapsible groups.
 * Non-tool parts and single tool invocations pass through as singles.
 * Empty text parts are skipped so they don't break runs of same-type tools.
 */
export function groupConsecutiveParts(parts: Part[]): GroupedItem[] {
  const result: GroupedItem[] = [];
  let i = 0;

  while (i < parts.length) {
    const part = parts[i];

    // Skip empty text parts entirely
    if (
      part.type === "text" &&
      "text" in part &&
      (part as { text: string }).text.trim() === ""
    ) {
      i++;
      continue;
    }

    // Non-tool parts pass through as singles
    if (!part.type.startsWith("tool-")) {
      result.push({ kind: "single", part, partIndex: i });
      i++;
      continue;
    }

    // Collect consecutive parts of the same tool type
    const toolType = part.type;
    const groupParts: Part[] = [part];
    const groupIndices: number[] = [i];
    let j = i + 1;

    while (j < parts.length) {
      const next = parts[j];
      if (next.type === toolType) {
        groupParts.push(next);
        groupIndices.push(j);
        j++;
      } else if (
        next.type === "text" &&
        "text" in next &&
        (next as { text: string }).text.trim() === ""
      ) {
        // Skip empty text parts within a potential group
        j++;
      } else {
        break;
      }
    }

    if (groupParts.length === 1) {
      result.push({ kind: "single", part, partIndex: i });
    } else {
      result.push({
        kind: "group",
        toolType,
        parts: groupParts,
        partIndices: groupIndices,
      });
    }

    i = j;
  }

  return result;
}
