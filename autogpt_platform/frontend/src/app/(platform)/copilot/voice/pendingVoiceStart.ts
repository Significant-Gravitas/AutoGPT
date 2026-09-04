/**
 * Carries "start voice mode" across the session-creation remount.
 *
 * `CopilotPage` keys the chat host by `sessionId ?? "new"`, so the first
 * message tears the host down and builds a new one. Voice mode has to be
 * asked for on the old mount and started on the new one; module scope is
 * the only place that survives, as it is for `pendingFirstSend`.
 */

let requested = false;

export function requestVoiceStart() {
  requested = true;
}

/** Consumes the request, so a later mount cannot start voice mode again. */
export function takeVoiceStart(): boolean {
  const wanted = requested;
  requested = false;
  return wanted;
}
