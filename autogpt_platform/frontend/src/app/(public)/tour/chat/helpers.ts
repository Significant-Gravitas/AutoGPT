import type { ArtifactRef } from "@/app/(platform)/copilot/store";
import type { TourMessage, TourPart, TourScenario } from "./script/types";

/** Build an ArtifactRef the real copilot ArtifactPanel can preview, copy and
 * download without a backend — the data: URI stands in for the workspace
 * download proxy the panel normally fetches from. */
export function buildTourArtifactRef(scenario: TourScenario): ArtifactRef {
  const { filename, markdown } = scenario.completionArtifact;
  return {
    id: `tour-${scenario.id}`,
    title: filename,
    mimeType: "text/markdown",
    sourceUrl: `data:text/markdown;charset=utf-8,${encodeURIComponent(markdown)}`,
    origin: "agent",
    sizeBytes: markdown.length,
  };
}

export function appendPartToLastMessage(
  messages: TourMessage[],
  part: TourPart,
): TourMessage[] {
  const next = messages.slice();
  const last = next[next.length - 1];
  next[next.length - 1] = { ...last, parts: [...last.parts, part] };
  return next;
}
