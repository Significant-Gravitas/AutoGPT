import {
  endOnboardingVoiceCall,
  finishOnboardingVoiceCall,
  getLatestOnboardingVoiceCall,
  startOnboardingVoiceCall,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { VoiceCallCaps } from "@/app/api/__generated__/models/voiceCallCaps";
import type { VoiceCallStartResponse } from "@/app/api/__generated__/models/voiceCallStartResponse";
import type { VoiceCallStatusResponse } from "@/app/api/__generated__/models/voiceCallStatusResponse";
import {
  type CallTicket,
  forgetCallTicket,
  isCallId,
  readCallTicket,
  saveCallTicket,
} from "@/services/onboarding/voice-call-ticket";
import { useEffect, useRef, useState } from "react";
import { trackInterview } from "../../../../tracking";
import { finishFailureOf, refusalOf, type StartFailure } from "../../helpers";
import {
  type CallAudio,
  connectCallAudio,
  createCallContext,
  openMicrophone,
} from "./audioGraph";
import {
  clearTimers,
  closeContext,
  closeSocket,
  type Connection,
  endCall,
  later,
  newConnection,
  type OpenCall,
  readStatus,
  release,
  sleep,
  stopTracks,
} from "./connection";
import {
  answerOf,
  type CallEnding,
  type CallPhase,
  type CallStartFailure,
  type CaptionLine,
  callStartFailureOf,
  ENDING_POLL_MS,
  ENDING_SLOW_POLL_MS,
  ENDING_WINDOW_MS,
  HANGUP_FALLBACK_MS,
  type HeldCall,
  OPENING_READ_TIMEOUT_MS,
  isRelaySocketUrl,
  type MicrophoneRefusal,
  microphoneRefusalOf,
  parseRelayMessage,
  REATTACH_RETRY_MS,
  REATTACH_WINDOW_MS,
  RESUMED_NOTICE_MS,
  secondsBetween,
  START_RETRIES,
  START_RETRY_MS,
  timeLeftCopy,
  voiceFinishOutcome,
  withCaption,
} from "./helpers";
import {
  CALL_IN_PROGRESS,
  CLOSE_NO_CALL,
  CLOSE_POLICY,
  CLOSE_REPLACED,
  CLOSE_TRY_AGAIN,
  type ClientMessage,
  type RelayMessage,
} from "./protocol";

type EndingAction =
  | "confirm"
  | "try_again"
  | "call_again"
  | "type_instead"
  | "skip";

interface Args {
  userId: string | null;
  // Whether the step offers calls, so a step opening again reads the
  // person's latest call.
  isOffered: boolean;
  onTypeInstead: (options: { remember: boolean }) => void;
  onLeave: (path: "A" | "B") => Promise<void>;
  onSkip: () => Promise<void>;
  onCallStarted: () => void;
  onMicrophoneRefused: (refusal: MicrophoneRefusal) => void;
  // The person has no call to pick up: the step decides what opens.
  onNoCall: () => Promise<void>;
}

const SETTLED_PHASES = new Set<CallPhase>(["idle", "ended"]);

export function useVoiceCall(args: Args) {
  const argsRef = useRef(args);
  useEffect(() => {
    argsRef.current = args;
  });
  const connRef = useRef<Connection>(newConnection());

  const [phase, setPhase] = useState<CallPhase>(
    args.isOffered ? "checking" : "idle",
  );
  const [captions, setCaptions] = useState<CaptionLine[]>([]);
  const [endsAt, setEndsAt] = useState<number | null>(null);
  const [now, setNow] = useState(0);
  const [turnsLeft, setTurnsLeft] = useState<number | null>(null);
  const [resumedAt, setResumedAt] = useState<number | null>(null);
  const [isWrappingUp, setIsWrappingUp] = useState(false);
  // The words have had no answer for the whole poll window: the person may
  // type instead while it goes on.
  const [isPastWindow, setIsPastWindow] = useState(false);
  const [isHangingUp, setIsHangingUp] = useState(false);
  const [startFailure, setStartFailure] = useState<CallStartFailure | null>(
    null,
  );
  const [ending, setEnding] = useState<CallEnding | null>(null);
  const [endReason, setEndReason] = useState<string | null>(null);
  const [finishFailure, setFinishFailure] = useState<StartFailure | null>(null);
  const [held, setHeld] = useState<HeldCall | null>(null);
  const [isLeaving, setIsLeaving] = useState(false);
  const [levels, setLevels] = useState<MediaStream | null>(null);

  useEffect(() => {
    const conn = connRef.current;
    return () => release(conn);
  }, []);

  // A step that opens again picks up the person's latest call, once.
  useEffect(() => {
    const conn = connRef.current;
    if (!args.isOffered || conn.hasChecked) return;
    conn.hasChecked = true;
    void checkLatestCall();
  });

  const isClockRunning = phase === "live" || phase === "reconnecting";
  useEffect(() => {
    if (!isClockRunning) return;
    const timer = setInterval(() => setNow(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, [isClockRunning]);

  // The step opening again reads the person's latest call before anything
  // typed, so turns typed before a call never hide the call that took the
  // interview: a live call is offered to rejoin; the record's word on the
  // interview decides next; then how the call ended.
  async function checkLatestCall() {
    const conn = connRef.current;
    const attempt = ++conn.attempt;
    setPhase("checking");
    let call: VoiceCallStatusResponse | null = null;
    const giveUp = new AbortController();
    const timer = setTimeout(() => giveUp.abort(), OPENING_READ_TIMEOUT_MS);
    try {
      const response = await getLatestOnboardingVoiceCall({
        signal: giveUp.signal,
      });
      if (response.status === 200) call = response.data;
    } catch {
      // No call yet, none that can be read, or no answer in time.
    } finally {
      clearTimeout(timer);
    }
    if (attempt !== conn.attempt) return;
    if (!call) {
      await argsRef.current.onNoCall();
      if (attempt === conn.attempt) setPhase("idle");
      return;
    }
    if (call.status === "connecting" || call.status === "live") {
      showHeld(call.call_id, "show");
      return;
    }
    conn.shownCallId = call.call_id;
    setEndReason(call.end_reason ?? null);
    const answer = answerOf(call);
    if (answer === "typed") {
      openTypedForLaterTake();
      return;
    }
    if (answer) {
      settle(answer, call);
      return;
    }
    setPhase("wrapping_up");
    void awaitEnding(call.call_id);
  }

  // A take the person made after the call owns the interview: the call is
  // no screen of its own, and the typed interview opens.
  function openTypedForLaterTake() {
    const conn = connRef.current;
    conn.poll += 1;
    resetView();
    setPhase("idle");
    argsRef.current.onTypeInstead({ remember: true });
  }

  // --- Starting a call ------------------------------------------------------

  function talk() {
    const context = createCallContext();
    void startCall(context);
  }

  async function startCall(context: AudioContext) {
    const conn = connRef.current;
    const attempt = beginAttempt();
    setHeld(null);
    setPhase("starting");
    let microphone: MediaStream;
    try {
      microphone = await openMicrophone();
    } catch (error) {
      closeContext(context);
      if (attempt === conn.attempt) refuseMicrophone(error);
      return;
    }
    // A later click took over: only one start may spend a call.
    if (attempt !== conn.attempt) {
      stopTracks(microphone);
      closeContext(context);
      return;
    }
    // The audio graph is built before the call is started, so a browser
    // that cannot run it never spends a call from the person's allowance.
    let audio: CallAudio;
    try {
      audio = await connectCallAudio(context, microphone);
    } catch {
      stopTracks(microphone);
      closeContext(context);
      if (attempt !== conn.attempt) return;
      setStartFailure({ kind: "failed" });
      setPhase("idle");
      return;
    }
    if (attempt !== conn.attempt) {
      audio.close();
      return;
    }
    let started: VoiceCallStartResponse;
    try {
      const response = await startOnboardingVoiceCall();
      if (
        response.status !== 200 ||
        !isCallId(response.data.call_id) ||
        !isRelaySocketUrl(response.data.relay_url)
      ) {
        throw new Error("No call started");
      }
      started = response.data;
    } catch (error) {
      audio.close();
      if (attempt === conn.attempt) refuseStart(error);
      return;
    }
    if (attempt !== conn.attempt) {
      audio.close();
      return;
    }
    argsRef.current.onCallStarted();
    attachAudio(audio);
    conn.call = {
      callId: started.call_id,
      relayUrl: started.relay_url,
      caps: started.caps,
      ticket: null,
      startedAt: null,
      turnsLeft: started.caps.max_agent_turns,
      turns: 0,
    };
    conn.shownCallId = started.call_id;
    conn.startRetries = 0;
    openSocket({ type: "start", token: started.token, leg: started.leg.kind });
  }

  function refuseMicrophone(error: unknown) {
    const refusal = microphoneRefusalOf(error);
    if (refusal === "blocked") trackInterview("voice_interview_mic_denied");
    setPhase("idle");
    argsRef.current.onMicrophoneRefused(refusal);
  }

  function refuseStart(error: unknown) {
    const refusal = refusalOf(error);
    const heldBy = refusal.heldBy;
    if (heldBy?.kind === "call_in_progress") {
      showHeld(heldBy.callId, "call");
      return;
    }
    if (heldBy?.kind === "closed" && heldBy.callId) {
      showTakenBy(heldBy.callId);
      return;
    }
    // Voice is off for this person, or their typed interview is over: the
    // typed step is the way on, without a word about it.
    if (refusal.status === 404 || heldBy?.kind === "closed") {
      setPhase("idle");
      argsRef.current.onTypeInstead({ remember: true });
      return;
    }
    setStartFailure(callStartFailureOf(refusal));
    setPhase("idle");
  }

  function attachAudio(audio: CallAudio) {
    const conn = connRef.current;
    conn.audio = audio;
    setLevels(audio.levels);
    // The person's audio goes to the relay only once the call is live.
    audio.onFrame((frame) => {
      const socket = conn.socket;
      if (conn.isReady && socket?.readyState === WebSocket.OPEN) {
        socket.send(frame);
      }
    });
  }

  // --- The call's socket ----------------------------------------------------

  function openSocket(first: ClientMessage) {
    const conn = connRef.current;
    const call = conn.call;
    if (!call) return;
    conn.relayError = null;
    let socket: WebSocket;
    try {
      socket = new WebSocket(call.relayUrl);
    } catch {
      onSocketClosed(1006, first);
      return;
    }
    socket.binaryType = "arraybuffer";
    conn.socket = socket;
    socket.onopen = () => {
      if (conn.socket === socket) socket.send(JSON.stringify(first));
    };
    socket.onmessage = (event: MessageEvent<string | ArrayBuffer>) => {
      if (conn.socket !== socket) return;
      if (typeof event.data === "string") {
        const message = parseRelayMessage(event.data);
        if (message) onMessage(message);
        return;
      }
      conn.audio?.play(event.data);
    };
    socket.onclose = (event: CloseEvent) => {
      if (conn.socket !== socket) return;
      conn.socket = null;
      onSocketClosed(event.code, first);
    };
  }

  function onMessage(message: RelayMessage) {
    const conn = connRef.current;
    switch (message.type) {
      case "ready":
        onReady(message.caps, message.resume, message.leg);
        return;
      case "agent_text":
        addCaption("otto", message.text);
        return;
      case "user_text":
        addCaption("you", message.text);
        return;
      case "interrupted":
        conn.audio?.flush();
        return;
      case "turn_complete":
        onTurnComplete(message.turns_left);
        return;
      case "wrap_up":
        setIsWrappingUp(true);
        return;
      case "resumed":
        onResumed(message.gap_s);
        return;
      case "ended":
        onEnded(message.reason, message.elapsed_s);
        return;
      case "error":
        conn.relayError = {
          error: message.error,
          callId: message.call_id ?? null,
        };
        return;
      default:
        return;
    }
  }

  function onReady(caps: VoiceCallCaps, ticket: string, leg: string) {
    const conn = connRef.current;
    const call = conn.call;
    if (!call) return;
    const startedAt = Date.now();
    call.ticket = ticket;
    call.caps = caps;
    call.startedAt = startedAt;
    call.turnsLeft = caps.max_agent_turns;
    conn.isReady = true;
    conn.dropDeadline = null;
    keepTicket(call);
    setNow(startedAt);
    setEndsAt(startedAt + caps.max_call_s * 1000);
    setTurnsLeft(caps.max_agent_turns);
    setPhase("live");
    trackInterview("interview_started", { mode: "voice", leg });
  }

  function addCaption(who: CaptionLine["who"], text: string) {
    const conn = connRef.current;
    const isNewTurn = conn.isNewTurn;
    conn.isNewTurn = false;
    setCaptions((lines) => withCaption(lines, who, text, isNewTurn));
  }

  function onTurnComplete(left: number) {
    const conn = connRef.current;
    conn.isNewTurn = true;
    const call = conn.call;
    if (call) {
      call.turns += 1;
      call.turnsLeft = left;
      keepTicket(call);
    }
    setTurnsLeft(left);
  }

  function onResumed(gapS: number) {
    const conn = connRef.current;
    conn.isReady = true;
    conn.dropDeadline = null;
    // A Hang up the drop may have swallowed is said again.
    if (conn.isHangingUp) {
      conn.socket?.send(
        JSON.stringify({ type: "hangup" } satisfies ClientMessage),
      );
    }
    setHeld(null);
    setResumedAt(Date.now());
    setNow(Date.now());
    setPhase("live");
    trackInterview("voice_interview_reconnected", { gap_s: gapS });
  }

  function onEnded(reason: string, elapsedS: number) {
    const conn = connRef.current;
    const call = conn.call;
    conn.call = null;
    conn.isReady = false;
    conn.isHangingUp = false;
    conn.dropDeadline = null;
    clearTimers(conn);
    finishAudio();
    forgetTicket();
    setIsHangingUp(false);
    setIsWrappingUp(false);
    setEndReason(reason);
    setPhase("wrapping_up");
    trackInterview("interview_ended", {
      mode: "voice",
      reason,
      elapsed_s: elapsedS,
      turns: call?.turns ?? 0,
    });
    if (call) void awaitEnding(call.callId);
  }

  function onSocketClosed(code: number, first: ClientMessage) {
    const conn = connRef.current;
    const call = conn.call;
    // The call ended, or this page let it go.
    if (!call) return;
    conn.isReady = false;
    // Another tab or device rejoined the call: it goes on there, and this
    // tab lets it go rather than taking it back.
    if (code === CLOSE_REPLACED) {
      dropCall();
      showHeld(call.callId, "show", { wentElsewhere: true });
      return;
    }
    if (!call.ticket) {
      onStartClosed(code, first);
      return;
    }
    if (code === CLOSE_NO_CALL || code === CLOSE_POLICY) {
      lose();
      return;
    }
    reattach();
  }

  // The socket closed before the call went live.
  function onStartClosed(code: number, first: ClientMessage) {
    const conn = connRef.current;
    const call = conn.call;
    const relayError = conn.relayError;
    if (relayError?.error === CALL_IN_PROGRESS && isCallId(relayError.callId)) {
      dropCall();
      showHeld(relayError.callId, "call");
      return;
    }
    if (code === CLOSE_TRY_AGAIN && conn.startRetries < START_RETRIES) {
      conn.startRetries += 1;
      later(
        conn,
        () => {
          if (conn.call === call) openSocket(first);
        },
        START_RETRY_MS,
      );
      return;
    }
    dropCall();
    setStartFailure(
      code === CLOSE_TRY_AGAIN
        ? { kind: "unavailable", retryAfterSeconds: null }
        : { kind: "failed" },
    );
    setPhase("idle");
  }

  // A live call's socket dropped: re-attach with the ticket while the relay
  // holds the call for it.
  function reattach() {
    const conn = connRef.current;
    const call = conn.call;
    if (!call?.ticket) return;
    if (conn.dropDeadline === null) openDropWindow(call);
    if (conn.dropDeadline !== null && Date.now() >= conn.dropDeadline) {
      lose();
      return;
    }
    const ticket = call.ticket;
    later(
      conn,
      () => {
        if (conn.call === call && !conn.isReady && conn.socket === null) {
          openSocket({ type: "resume", ticket });
        }
      },
      REATTACH_RETRY_MS,
    );
  }

  function openDropWindow(call: OpenCall) {
    const conn = connRef.current;
    const drop = ++conn.drop;
    conn.droppedAt = Date.now();
    conn.dropDeadline = Date.now() + REATTACH_WINDOW_MS;
    setResumedAt(null);
    setPhase("reconnecting");
    later(
      conn,
      () => {
        if (conn.drop === drop && conn.call === call && !conn.isReady) lose();
      },
      REATTACH_WINDOW_MS,
    );
  }

  // The call could not be re-attached. Its words may still be taken, so the
  // step waits for their answer before it offers anything else.
  function lose() {
    const conn = connRef.current;
    const call = conn.call;
    if (!call) return;
    conn.lostCall = {
      turns: call.turns,
      startedAt: call.startedAt,
      droppedAt: conn.droppedAt ?? Date.now(),
    };
    conn.call = null;
    conn.isReady = false;
    conn.isHangingUp = false;
    conn.dropDeadline = null;
    clearTimers(conn);
    closeSocket(conn);
    finishAudio();
    forgetTicket();
    setIsHangingUp(false);
    setIsWrappingUp(false);
    setPhase("saving");
    void awaitEnding(call.callId);
  }

  // --- What became of the call's words ------------------------------------

  // Reads the call until its words have an answer. Past the window the
  // screen offers "Type instead" beside the wait, and the read goes on until
  // the person takes it: from then on the server settles who owns the
  // interview, not this screen.
  async function awaitEnding(callId: string) {
    const conn = connRef.current;
    const poll = ++conn.poll;
    conn.shownCallId = callId;
    setIsPastWindow(false);
    const windowEndsAt = Date.now() + ENDING_WINDOW_MS;
    for (;;) {
      const call = await readStatus(callId);
      if (poll !== conn.poll) return;
      if (call === "missing") {
        // A call named as the interview's take whose record has expired:
        // its hand-off ran long ago.
        settle(
          conn.taken ? { kind: "team", path: "A" } : { kind: "missing" },
          null,
        );
        return;
      }
      const answer = call ? answerOf(call, { taken: conn.taken }) : null;
      if (answer === "typed") {
        openTypedForLaterTake();
        return;
      }
      if (answer && call) {
        settle(answer, call);
        return;
      }
      const isPastWindow = Date.now() >= windowEndsAt;
      if (isPastWindow) setIsPastWindow(true);
      await sleep(isPastWindow ? ENDING_SLOW_POLL_MS : ENDING_POLL_MS);
      if (poll !== conn.poll) return;
    }
  }

  function settle(next: CallEnding, call: VoiceCallStatusResponse | null) {
    const conn = connRef.current;
    const lost = conn.lostCall;
    if (lost && call?.end_reason) {
      conn.lostCall = null;
      trackInterview("interview_ended", {
        mode: "voice",
        reason: call.end_reason,
        elapsed_s: lost.startedAt
          ? secondsBetween(lost.startedAt, lost.droppedAt)
          : null,
        turns: lost.turns,
      });
    }
    if (call?.end_reason) setEndReason(call.end_reason);
    setEnding(next);
    setFinishFailure(null);
    setPhase("ended");
    trackInterview(
      "ending_shown",
      { mode: "voice", ending: next.kind },
      {
        oncePerTabKey: `voice_ending_shown_${conn.shownCallId ?? "none"}_${next.kind}`,
      },
    );
  }

  // --- The person's choices -------------------------------------------------

  // The person's own close. The microphone stops at once; a re-attach says
  // it again; and a relay that has not ended the call a few seconds later is
  // asked to through the REST API.
  function hangUp() {
    const conn = connRef.current;
    const call = conn.call;
    const socket = conn.socket;
    if (!call || !conn.isReady || socket?.readyState !== WebSocket.OPEN) return;
    conn.isHangingUp = true;
    setIsHangingUp(true);
    conn.audio?.mute();
    socket.send(JSON.stringify({ type: "hangup" } satisfies ClientMessage));
    later(
      conn,
      () => {
        if (conn.call === call) {
          void endOnboardingVoiceCall(call.callId).catch(() => undefined);
        }
      },
      HANGUP_FALLBACK_MS,
    );
  }

  // Leaves the call for the typed interview. A live call is ended first, so
  // its words are kept as a call cut off rather than one the person closed,
  // and the typed interview opens.
  async function typeInstead() {
    const conn = connRef.current;
    if (phase === "ended") trackEndingAction("type_instead");
    const attempt = ++conn.attempt;
    conn.poll += 1;
    // A finish waiting on the server is let go: the typed interview takes
    // over from here.
    conn.isLeaving = false;
    setIsLeaving(false);
    const call = conn.call;
    if (call) {
      trackInterview("interview_ended", {
        mode: "voice",
        reason: "type_instead",
        elapsed_s: call.startedAt
          ? secondsBetween(call.startedAt, Date.now())
          : null,
        turns: call.turns,
      });
      dropCall();
      forgetTicket();
      setPhase("ending_call");
      await endCall(call.callId);
      if (attempt !== conn.attempt) return;
    }
    resetView();
    setPhase("idle");
    argsRef.current.onTypeInstead({ remember: true });
  }

  function callAgain() {
    trackEndingAction("call_again");
    talk();
  }

  async function skip() {
    trackEndingAction("skip");
    await argsRef.current.onSkip();
  }

  async function meetTeam() {
    const conn = connRef.current;
    if (conn.isLeaving || !ending) return;
    trackEndingAction("confirm");
    if (ending.kind === "team" && ending.path) {
      conn.isLeaving = true;
      setIsLeaving(true);
      await argsRef.current.onLeave(ending.path);
      return;
    }
    if (conn.shownCallId) await finishCall(conn.shownCallId);
  }

  async function retryFinish() {
    const conn = connRef.current;
    if (conn.isLeaving || !conn.shownCallId) return;
    trackEndingAction("try_again");
    await finishCall(conn.shownCallId);
  }

  // The person's word on the call's take: hand off what it kept, or answer
  // the hand-off as it stands, and read the answer for a call. A finish the
  // server cannot take yet (409: the call's words, or a later call of the
  // person's, are not settled) is waited out and asked again, never shown as
  // an error.
  async function finishCall(callId: string) {
    const conn = connRef.current;
    const attempt = ++conn.attempt;
    const poll = ++conn.poll;
    conn.isLeaving = true;
    setIsLeaving(true);
    setFinishFailure(null);
    const windowEndsAt = Date.now() + ENDING_WINDOW_MS;
    let outcome: ReturnType<typeof voiceFinishOutcome> = null;
    for (;;) {
      try {
        const response = await finishOnboardingVoiceCall(callId);
        outcome =
          response.status === 200 ? voiceFinishOutcome(response.data) : null;
        break;
      } catch (error) {
        // The step went away, or the person typed instead, meanwhile.
        if (attempt !== conn.attempt || poll !== conn.poll) return;
        if (refusalOf(error).status !== 409) {
          stopFinishing(finishFailureOf(error));
          return;
        }
      }
      setPhase("finishing");
      const isPastWindow = Date.now() >= windowEndsAt;
      setIsPastWindow(isPastWindow);
      await sleep(isPastWindow ? ENDING_SLOW_POLL_MS : ENDING_POLL_MS);
      if (attempt !== conn.attempt || poll !== conn.poll) return;
    }
    if (attempt !== conn.attempt || poll !== conn.poll) return;
    setIsPastWindow(false);
    if (outcome === "superseded") {
      // Another take owns the interview, and the call's record says whose:
      // the person's typing opens the typed interview; any other leads on.
      const call = await readStatus(callId);
      if (attempt !== conn.attempt || poll !== conn.poll) return;
      if (call === null) {
        stopFinishing({ kind: "failed" });
        return;
      }
      if (call !== "missing" && answerOf(call) === "typed") {
        conn.isLeaving = false;
        setIsLeaving(false);
        openTypedForLaterTake();
        return;
      }
      outcome = "A";
    }
    if (outcome === "A") {
      await argsRef.current.onLeave("A");
      return;
    }
    conn.isLeaving = false;
    setIsLeaving(false);
    if (outcome === "no_words") {
      settle({ kind: "no_words" }, null);
      return;
    }
    if (outcome === "insufficient") {
      settle({ kind: "insufficient" }, null);
      return;
    }
    if (outcome === "unsaved") {
      settle({ kind: "unsaved", reason: endReason }, null);
      return;
    }
    stopFinishing({ kind: "failed" });
  }

  // A finish that cannot go on: back to the call's ending, saying why.
  function stopFinishing(failure: StartFailure) {
    const conn = connRef.current;
    conn.isLeaving = false;
    setIsLeaving(false);
    setIsPastWindow(false);
    setPhase("ended");
    setFinishFailure(failure);
  }

  function trackEndingAction(action: EndingAction) {
    if (!ending) return;
    trackInterview("ending_action", {
      mode: "voice",
      ending: ending.kind,
      action,
    });
  }

  // --- A call that holds the interview -------------------------------------

  // A live call of the person's holds the interview: rejoin it with the
  // ticket this tab kept, or end it.
  function showHeld(
    callId: string,
    then: HeldCall["then"],
    { wentElsewhere = false }: { wentElsewhere?: boolean } = {},
  ) {
    const conn = connRef.current;
    conn.hasChecked = true;
    conn.attempt += 1;
    conn.poll += 1;
    const userId = argsRef.current.userId;
    setHeld({
      callId,
      canRejoin: !!userId && readCallTicket(userId, callId) !== null,
      then,
      wentElsewhere,
    });
    setPhase("held");
  }

  // The server named this call as the one that took the interview: show its
  // ending.
  function showTakenBy(callId: string) {
    const conn = connRef.current;
    conn.hasChecked = true;
    conn.attempt += 1;
    resetView();
    conn.taken = true;
    setPhase("checking");
    void awaitEnding(callId);
  }

  function rejoin() {
    const userId = argsRef.current.userId;
    const kept = held && userId ? readCallTicket(userId, held.callId) : null;
    if (!held) return;
    if (!kept) {
      setHeld({ ...held, canRejoin: false });
      return;
    }
    const context = createCallContext();
    void rejoinCall(context, kept);
  }

  async function rejoinCall(context: AudioContext, kept: CallTicket) {
    const conn = connRef.current;
    const attempt = beginAttempt();
    setPhase("reconnecting");
    // The socket to rejoin on is the server's word, read now; the call may
    // also have ended meanwhile.
    const call = await readStatus(kept.callId);
    if (attempt !== conn.attempt) {
      closeContext(context);
      return;
    }
    const isLive =
      call !== null &&
      call !== "missing" &&
      (call.status === "connecting" || call.status === "live");
    const relayUrl = isLive ? relayUrlOf(call) : null;
    if (!relayUrl) {
      closeContext(context);
      if (call === null) {
        setPhase("held");
        return;
      }
      setHeld(null);
      setPhase("wrapping_up");
      void awaitEnding(kept.callId);
      return;
    }
    let microphone: MediaStream;
    try {
      microphone = await openMicrophone();
    } catch (error) {
      closeContext(context);
      if (attempt === conn.attempt) refuseMicrophone(error);
      return;
    }
    let audio: CallAudio;
    try {
      audio = await connectCallAudio(context, microphone);
    } catch {
      stopTracks(microphone);
      closeContext(context);
      if (attempt === conn.attempt) setPhase("held");
      return;
    }
    if (attempt !== conn.attempt) {
      audio.close();
      return;
    }
    attachAudio(audio);
    const rejoined: OpenCall = {
      callId: kept.callId,
      relayUrl,
      caps: kept.caps,
      ticket: kept.ticket,
      startedAt: kept.startedAt,
      turnsLeft: kept.turnsLeft,
      turns: Math.max(0, kept.caps.max_agent_turns - kept.turnsLeft),
    };
    conn.call = rejoined;
    conn.shownCallId = kept.callId;
    setNow(Date.now());
    setEndsAt(kept.startedAt + kept.caps.max_call_s * 1000);
    setTurnsLeft(kept.turnsLeft);
    openDropWindow(rejoined);
    openSocket({ type: "resume", ticket: kept.ticket });
  }

  function endHeldCall() {
    if (!held) return;
    // A new call follows from this click, which is the gesture its audio
    // needs.
    const context = held.then === "call" ? createCallContext() : null;
    void endHeldCallThen(held, context);
  }

  async function endHeldCallThen(
    current: HeldCall,
    context: AudioContext | null,
  ) {
    const conn = connRef.current;
    const attempt = ++conn.attempt;
    setPhase("ending_call");
    await endCall(current.callId);
    if (attempt !== conn.attempt) {
      if (context) closeContext(context);
      return;
    }
    const userId = argsRef.current.userId;
    if (userId && readCallTicket(userId, current.callId)) forgetTicket();
    setHeld(null);
    if (current.then === "call" && context) {
      void startCall(context);
      return;
    }
    if (current.then === "type") {
      setPhase("idle");
      argsRef.current.onTypeInstead({ remember: false });
      return;
    }
    setPhase("wrapping_up");
    void awaitEnding(current.callId);
  }

  // --- Plumbing -------------------------------------------------------------

  function beginAttempt() {
    const conn = connRef.current;
    conn.attempt += 1;
    conn.poll += 1;
    conn.taken = false;
    conn.isNewTurn = true;
    conn.isHangingUp = false;
    conn.lostCall = null;
    resetView();
    return conn.attempt;
  }

  function resetView() {
    setCaptions([]);
    setEndsAt(null);
    setTurnsLeft(null);
    setResumedAt(null);
    setIsWrappingUp(false);
    setIsPastWindow(false);
    setIsHangingUp(false);
    setStartFailure(null);
    setEnding(null);
    setEndReason(null);
    setFinishFailure(null);
    setLevels(null);
  }

  // Lets the page's own call go without telling the relay anything: a call
  // ended elsewhere, or about to be.
  function dropCall() {
    const conn = connRef.current;
    conn.call = null;
    conn.isReady = false;
    conn.isHangingUp = false;
    conn.dropDeadline = null;
    clearTimers(conn);
    closeSocket(conn);
    conn.audio?.close();
    conn.audio = null;
    setLevels(null);
  }

  // Otto's last words play out; the microphone stops now.
  function finishAudio() {
    const conn = connRef.current;
    conn.audio?.finish();
    conn.audio = null;
    setLevels(null);
  }

  function keepTicket(call: OpenCall) {
    const userId = argsRef.current.userId;
    if (!userId || !call.ticket || call.startedAt === null) return;
    saveCallTicket(userId, {
      callId: call.callId,
      ticket: call.ticket,
      caps: call.caps,
      startedAt: call.startedAt,
      turnsLeft: call.turnsLeft,
    });
  }

  function forgetTicket() {
    const userId = argsRef.current.userId;
    if (userId) forgetCallTicket(userId);
  }

  const isBusy = !SETTLED_PHASES.has(phase);
  const isOver = phase === "ended" && ending?.kind === "team";

  return {
    phase,
    captions,
    timeLeft: endsAt !== null ? timeLeftCopy(endsAt - now) : null,
    turnsLeft,
    isWrappingUp,
    isPastWindow,
    wasResumed:
      phase === "live" &&
      resumedAt !== null &&
      now - resumedAt < RESUMED_NOTICE_MS,
    isHangingUp,
    startFailure,
    ending,
    endReason,
    finishFailure,
    isLeaving,
    held,
    levels,
    // A call is under way or its words have no answer yet: the step keeps
    // its Skip and Back out of the way.
    isBusy,
    // The call took the interview.
    isOver,
    talk,
    rejoin,
    endHeldCall,
    hangUp,
    typeInstead,
    callAgain,
    skip,
    meetTeam,
    retryFinish,
    showHeld,
    showTakenBy,
  };
}

export type VoiceCallState = ReturnType<typeof useVoiceCall>;

// The relay socket the server names for a live call, when it is one.
function relayUrlOf(call: VoiceCallStatusResponse) {
  return isRelaySocketUrl(call.relay_url) ? call.relay_url : null;
}
