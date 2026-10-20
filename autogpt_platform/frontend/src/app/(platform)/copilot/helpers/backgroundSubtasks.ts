import { asObject } from "../components/ToolChain/resultHelpers";

export function backgroundRefreshInterval(
  messages: unknown[],
  now = Date.now(),
) {
  const pending = new Map<string, number>();
  let followup: string | null = null;
  for (const raw of messages) {
    const message = asObject(raw);
    if (!message) continue;
    const metadata = asObject(message.metadata);
    if (message.role === "tool") {
      const output = asObject(message.content);
      if (
        typeof output?.background_task_id === "string" &&
        typeof output.background_expires_at === "string"
      ) {
        pending.set(
          output.background_task_id,
          Date.parse(output.background_expires_at),
        );
      }
    }
    if (typeof metadata?.background_task_notice === "string") {
      followup = metadata.background_task_notice;
    }
    if (message.role === "assistant" && followup) {
      pending.delete(followup);
      followup = null;
    }
  }
  return [...pending.values()].some((expires) => expires > now) ? 3000 : false;
}

export function backgroundNotice(metadata: Record<string, unknown> | null) {
  return metadata?.background_task_stage === "followup" &&
    typeof metadata.display_content === "string"
    ? metadata.display_content
    : null;
}
