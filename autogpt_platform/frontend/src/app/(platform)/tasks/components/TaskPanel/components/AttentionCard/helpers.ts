import type { Ask } from "@/app/api/__generated__/models/ask";
import type { Blocker } from "@/app/api/__generated__/models/blocker";
import type { Criterion } from "@/app/api/__generated__/models/criterion";
import { actionLabel } from "../../../../data/helpers";

/** Checks the work didn't meet, so accepting it is a choice made knowingly. */
export function unmetChecks(doneWhen: Criterion[] | undefined) {
  return (doneWhen ?? [])
    .filter((c) => !c.status || c.status === "unmet" || c.status === "pending")
    .map((c) => c.text);
}

/** The worker's pick first, so it is the primary button. */
export function orderOptions(ask: Ask) {
  const options = ask.options ?? [];
  const pick = ask.default_option;
  if (!pick || !options.includes(pick)) return options;
  return [pick, ...options.filter((o) => o !== pick)];
}

const RETRY = "retry_failed_part";

export interface CardActions {
  /** Action keys posted to the actions route, Retry first. */
  actions: string[];
  /** The worker asked for a file. */
  wantsUpload: boolean;
}

/** A failure's ways out: Retry always, first; an upload ask becomes the
 *  Upload file button rather than an action the route can't run. */
export function failureActions(nextActions: string[] | undefined): CardActions {
  const all = nextActions ?? [];
  const wantsUpload = all.some(isUpload);
  const rest = all.filter((a) => !isUpload(a) && !isRetry(a));
  const retry = all.find(isRetry) ?? RETRY;
  return { actions: [retry, ...rest], wantsUpload };
}

/** The ways out of a wait that isn't a failure, as the worker listed them. */
export function waitActions(nextActions: string[] | undefined): CardActions {
  const all = nextActions ?? [];
  return {
    actions: all.filter((a) => !isUpload(a)),
    wantsUpload: all.some(isUpload),
  };
}

export function cardActionLabel(action: string) {
  return isRetry(action) ? "Retry" : actionLabel(action);
}

function isRetry(action: string) {
  return /^retry/i.test(action.trim());
}

function isUpload(action: string) {
  return /upload|attach/i.test(action);
}

interface AttentionArgs {
  failed: boolean;
  blocker: Blocker | null;
  waitsOnYou: boolean;
  awaitingAccept: boolean;
}

/** The card's buttons: a failure's ways out, a wait's own ones, or Accept
 *  for finished work. A question or hire cards carry their own controls. */
export function attentionActions({
  failed,
  blocker,
  waitsOnYou,
  awaitingAccept,
}: AttentionArgs): CardActions {
  if (failed) return failureActions(blocker?.next_actions);
  if (blocker) return waitActions(blocker.next_actions);
  if (!waitsOnYou && awaitingAccept)
    return { actions: ["accept"], wantsUpload: false };
  return { actions: [], wantsUpload: false };
}
