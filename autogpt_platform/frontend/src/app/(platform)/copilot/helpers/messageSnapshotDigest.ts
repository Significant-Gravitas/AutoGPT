import type { UIMessage } from "ai";
import { getSessionActivity } from "../components/WorkspaceFileCards/helpers";

/**
 * What the subscribers of a message snapshot can read from it, as one
 * comparable string: the runs and schedules `useSessionActivity` mines, and
 * the parts `useSessionFiles` finds files in — file parts, and text that holds
 * a `workspace://` URI — with the id and role of their message. Two snapshots
 * with the same digest look the same to every subscriber, so a streamed token
 * that only adds plain text does not have to wake them. It may differ where
 * they would not, which only costs an extra update; it never matches where
 * they would differ.
 */
export function getMessageSnapshotDigest(messages: UIMessage[]): string {
  const fileSources: unknown[] = [];
  for (const message of messages) {
    const sources = message.parts.flatMap((part): unknown[] => {
      if (part.type === "file") {
        return [[part.url, part.filename ?? null, part.mediaType]];
      }
      if (part.type === "text" && part.text.includes("workspace://")) {
        return [part.text];
      }
      return [];
    });
    if (sources.length > 0) {
      fileSources.push([message.id, message.role, sources]);
    }
  }
  return JSON.stringify([fileSources, getSessionActivity(messages)]);
}
