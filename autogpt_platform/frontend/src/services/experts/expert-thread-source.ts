import { trackFunnel } from "@/services/experts/experts-analytics";

// Where the next expert thread was started from, for `expert_thread_created`'s
// `thread_source`.
// Set just before a send that will create the thread and taken by the
// session's creation, so it rides on exactly one event; a send that fails
// before creating a thread clears it.

export type ExpertThreadSource = "suggested_task";

let nextSource: ExpertThreadSource | null = null;

export function markExpertThreadSource(source: ExpertThreadSource | null) {
  nextSource = source;
}

export function takeExpertThreadSource() {
  const source = nextSource;
  nextSource = null;
  return source;
}

export function trackExpertThreadCreated(
  expertID: string,
  source: ExpertThreadSource | null,
) {
  trackFunnel(
    "expert_thread_created",
    // Not `source`: that base property names the emitter (`web`).
    source
      ? { expert_id: expertID, thread_source: source }
      : { expert_id: expertID },
  );
}
