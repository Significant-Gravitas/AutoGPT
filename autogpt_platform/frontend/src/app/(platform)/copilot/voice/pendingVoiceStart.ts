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

/**
 * True while voice mode is listening or speaking. Read by the chat
 * transport, which sits outside this feature and has no other way to know
 * the reply will be spoken.
 */
let voiceTurnActive = false;

export function setVoiceTurnActive(active: boolean) {
  voiceTurnActive = active;
}

export function isVoiceTurn(): boolean {
  return voiceTurnActive;
}
