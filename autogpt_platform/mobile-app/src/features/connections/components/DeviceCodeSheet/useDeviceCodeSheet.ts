import { useMutation, useQuery } from "@tanstack/react-query";
import * as Clipboard from "expo-clipboard";
import * as WebBrowser from "expo-web-browser";
import { useState } from "react";

import { ApiError } from "@/lib/api/client";
import {
  cancelCodexLogin,
  finishProviderLogin,
  getCodexLoginStatus,
  startProviderLogin,
  type CredentialMeta,
} from "@/lib/api/integrations";
import { sessionGeneration } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

import { deviceLoginID, parseDeviceLogin } from "../../deviceLogin";
import { CODEX_PROVIDER } from "../../helpers";
import { rememberCredential } from "../../oauth";

const POLL_MS = 3000;
const CANT_START = "ChatGPT sign-in can't start right now. Try again later.";
const DIDNT_FINISH = "ChatGPT sign-in didn't finish.";

// The backend runs one ChatGPT sign-in per user at a time, so one is tracked.
let pendingLoginID: string | null = null;
// Counts closes, so a sign-in that starts after its sheet closed is ended at once.
let closeCount = 0;

function settle(loginID: string) {
  if (pendingLoginID === loginID) pendingLoginID = null;
}

/** Ends the sign-in still waiting for its code, so the next one can start. */
function cancelPendingDeviceLogin() {
  const loginID = pendingLoginID;
  pendingLoginID = null;
  if (loginID) void cancelCodexLogin(loginID).catch(() => undefined);
}

/** Closing a sheet: ends its sign-in, even one still starting. */
export function closeDeviceLogin() {
  closeCount += 1;
  cancelPendingDeviceLogin();
}

/** Ends a sign-in left open, so the backend lets a new one start. */
async function endStaleLogin() {
  const loginID = pendingLoginID;
  pendingLoginID = null;
  if (loginID) await cancelCodexLogin(loginID).catch(() => undefined);
}

async function startLogin(credentialID: string | undefined) {
  await endStaleLogin();
  const login = await startProviderLogin(CODEX_PROVIDER, { scopes: [], credentialID });
  const device = parseDeviceLogin(login.loginURL);
  if (!device) {
    // Nothing to show, but the backend holds it open: end it now.
    const loginID = deviceLoginID(login.loginURL, login.cancelURL);
    if (loginID) await cancelCodexLogin(loginID).catch(() => undefined);
  }
  return { state: login.state, device };
}

async function pollLogin(loginID: string) {
  try {
    const result = await getCodexLoginStatus(loginID);
    if (result.status !== "pending") settle(loginID);
    return result;
  } catch (error) {
    // Gone: it expired, or was ended elsewhere.
    if (error instanceof ApiError && error.status === 404) {
      settle(loginID);
      return { status: "failed" as const, error: null };
    }
    throw error;
  }
}

function isPaywall(error: unknown) {
  return error instanceof ApiError && error.status === 402;
}

interface Args {
  /** An account to sign in again in place. */
  credentialID?: string;
  onSaved: (credential: CredentialMeta) => void;
  onClose: () => void;
  onPlanRequired: () => void;
}

export function useDeviceCodeSheet({ credentialID, onSaved, onClose, onPlanRequired }: Args) {
  const [isCopied, setIsCopied] = useState(false);
  // Finished after a sign-out: the account isn't this session's to show.
  const [isAbandoned, setIsAbandoned] = useState(false);

  const start = useMutation({
    onMutate() {
      return closeCount;
    },
    mutationFn: () => startLogin(credentialID),
    onSuccess({ device }, _variables, openedAt) {
      if (!device) return;
      pendingLoginID = device.loginID;
      if (closeCount !== openedAt) cancelPendingDeviceLogin();
    },
    onError(error) {
      if (isPaywall(error)) onPlanRequired();
    },
  });
  const device = start.data?.device ?? null;

  const status = useQuery({
    queryKey: ["integrations", "codex", "device-login", device?.loginID ?? ""],
    queryFn: () => pollLogin(device?.loginID ?? ""),
    enabled: device !== null,
    refetchInterval: (query) =>
      (query.state.data?.status ?? "pending") === "pending" ? POLL_MS : false,
    gcTime: 0,
  });
  const loginStatus = status.data?.status ?? "pending";

  const complete = useMutation({
    onMutate() {
      return sessionGeneration();
    },
    mutationFn: ({ loginID, state }: { loginID: string; state: string }) =>
      finishProviderLogin(CODEX_PROVIDER, loginID, state),
    onSuccess(credential, _variables, startedIn) {
      if (startedIn !== sessionGeneration()) {
        setIsAbandoned(true);
        return;
      }
      rememberCredential(credential);
      onSaved(credential);
    },
    onError(error) {
      if (isPaywall(error)) onPlanRequired();
    },
  });

  useMountEffect(() => {
    start.mutate();
  });

  function startError() {
    if (start.isError) {
      const { error } = start;
      return error instanceof ApiError && error.status === 409 ? error.message : CANT_START;
    }
    return start.isSuccess && !device ? CANT_START : null;
  }

  function failure() {
    if (loginStatus === "failed" || loginStatus === "canceled" || isAbandoned) {
      return DIDNT_FINISH;
    }
    if (complete.isError) return "Couldn't connect ChatGPT. Try again.";
    return null;
  }

  function finish() {
    if (!device || !start.data || !complete.isIdle) return;
    complete.mutate({ loginID: device.loginID, state: start.data.state });
  }

  function retry() {
    setIsCopied(false);
    setIsAbandoned(false);
    complete.reset();
    start.mutate();
  }

  async function copyCode() {
    if (!device) return;
    await Clipboard.setStringAsync(device.userCode).catch(() => undefined);
    setIsCopied(true);
  }

  function openChatGPT() {
    if (device) void WebBrowser.openBrowserAsync(device.verificationURL).catch(() => undefined);
  }

  function close() {
    closeDeviceLogin();
    onClose();
  }

  const cantStart = startError();
  return {
    userCode: device?.userCode ?? null,
    isStarting: start.isPending || start.isIdle,
    cantStart,
    failure: cantStart ? null : failure(),
    isFinishing: complete.isPending,
    shouldFinish: loginStatus === "completed" && complete.isIdle,
    isCopied,
    copyCode,
    openChatGPT,
    finish,
    retry,
    close,
  };
}
