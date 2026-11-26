import { useMutation, useQuery } from "@tanstack/react-query";
import * as Clipboard from "expo-clipboard";
import * as WebBrowser from "expo-web-browser";
import { useRef, useState } from "react";

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

import { parseDeviceLogin } from "../../deviceLogin";
import { CODEX_PROVIDER } from "../../helpers";
import { rememberCredential } from "../../oauth";

const POLL_MS = 3000;
const CANT_START = "ChatGPT sign-in can't start right now. Try again later.";
const DIDNT_FINISH = "ChatGPT sign-in didn't finish.";

// The backend runs one ChatGPT sign-in per user at a time, so one is tracked.
let pendingLoginID: string | null = null;

function settle(loginID: string) {
  if (pendingLoginID === loginID) pendingLoginID = null;
}

/** Ends the sign-in still waiting for its code, so the next one can start. */
export function cancelPendingDeviceLogin() {
  const loginID = pendingLoginID;
  pendingLoginID = null;
  if (loginID) void cancelCodexLogin(loginID).catch(() => undefined);
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
  // A sign-in that starts after the sheet closed is ended at once.
  const isClosed = useRef(false);

  const start = useMutation({
    async mutationFn() {
      const login = await startProviderLogin(CODEX_PROVIDER, { scopes: [], credentialID });
      return { state: login.state, device: parseDeviceLogin(login.loginURL) };
    },
    onSuccess({ device }) {
      if (!device) return;
      pendingLoginID = device.loginID;
      if (isClosed.current) cancelPendingDeviceLogin();
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
      // Signed out meanwhile: the account belongs to whoever started this.
      if (startedIn !== sessionGeneration()) return;
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
    if (loginStatus === "failed" || loginStatus === "canceled") return DIDNT_FINISH;
    if (complete.isError) return "Couldn't connect ChatGPT. Try again.";
    return null;
  }

  function finish() {
    if (!device || !start.data || !complete.isIdle) return;
    complete.mutate({ loginID: device.loginID, state: start.data.state });
  }

  function retry() {
    cancelPendingDeviceLogin();
    setIsCopied(false);
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
    isClosed.current = true;
    if (loginStatus === "pending") cancelPendingDeviceLogin();
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
