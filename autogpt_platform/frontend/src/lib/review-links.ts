import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";

/**
 * Deep link to wherever a pending human review can be acted on.
 *
 * Lives here rather than next to a component so review surfaces (copilot
 * home, thread, library) share one definition of these routes instead of
 * each embedding path literals.
 *
 * Both ids come from the backend's review enrichment: `session_id` is set
 * for every CoPilot `run_block` review, `library_agent_id` for reviews
 * raised by a real graph execution.
 *
 * The library route selects a run via `activeTab`/`activeItem` — those are
 * the params `NewAgentLibraryView` actually parses, so anything else lands
 * on the agent page without opening the run.
 */
export function getReviewLink(review: PendingHumanReviewModel): string {
  if (review.session_id) {
    return `/copilot?sessionId=${encodeURIComponent(review.session_id)}`;
  }
  if (review.library_agent_id) {
    return (
      `/library/agents/${encodeURIComponent(review.library_agent_id)}` +
      `?activeTab=runs&activeItem=${encodeURIComponent(review.graph_exec_id)}`
    );
  }
  return "/library";
}
