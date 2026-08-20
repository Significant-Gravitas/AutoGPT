"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV1ListCredentialsQueryKey,
  postV1InitiateDeviceCodeOauthFlow,
  postV1PollDeviceCodeOauthFlowForCompletion,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { toast } from "@/components/molecules/Toast/use-toast";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

interface Args {
  provider: string;
  /**
   * Receives the credential the poll returned, so a caller that has to select
   * it (the builder's credentials input) doesn't have to re-fetch and guess
   * which of the provider's credentials is the new one.
   */
  onSuccess: (credentials?: CredentialsMetaResponse) => void;
}

type Phase = "idle" | "awaiting_user" | "polling" | "done" | "error";

const MAX_CONSECUTIVE_POLL_FAILURES = 3;

export function useDeviceAuthConnect({ provider, onSuccess }: Args) {
  const queryClient = useQueryClient();
  const [phase, setPhase] = useState<Phase>("idle");
  const [userCode, setUserCode] = useState("");
  const [verificationUrl, setVerificationUrl] = useState("");

  const isUnmountedRef = useRef(false);
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intervalRef = useRef(5);
  // Identifies the live loop. Clearing the timeout cannot cancel a request
  // already in flight, so a cancel mid-round-trip would let the resolved poll
  // re-arm the loop — and two overlapping connect() calls would leave two live
  // loops with only the newest in pollingRef. Each loop captures its id and
  // stops if it is no longer the current one.
  const runIdRef = useRef(0);
  // A flow stays open ~10 minutes. One blip in that window must not end it.
  const consecutiveFailuresRef = useRef(0);

  useEffect(() => {
    // Reset on mount: the ref survives a remount (StrictMode, or reopening the
    // dialog), and a stale  would make every callback below no-op.
    isUnmountedRef.current = false;
    return () => {
      isUnmountedRef.current = true;
      if (pollingRef.current) clearTimeout(pollingRef.current);
    };
  }, []);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearTimeout(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const poll = useCallback(
    async (token: string, runId: number) => {
      if (isUnmountedRef.current || runId !== runIdRef.current) return;

      try {
        const response = await postV1PollDeviceCodeOauthFlowForCompletion(
          provider,
          { state_token: token },
        );

        if (isUnmountedRef.current || runId !== runIdRef.current) return;

        if (response.status !== 200) {
          throw new Error("Device auth poll failed");
        }
        consecutiveFailuresRef.current = 0;
        const { status, credentials } = response.data;

        if (status === "approved") {
          setPhase("done");
          stopPolling();
          toast({ title: "Connected via device auth", variant: "success" });
          await queryClient.invalidateQueries({
            queryKey: getGetV1ListCredentialsQueryKey(),
          });
          onSuccess(credentials ?? undefined);
          return;
        }

        if (status === "slow_down") {
          intervalRef.current = Math.min(intervalRef.current + 5, 30);
        }

        if (status === "denied" || status === "expired") {
          setPhase("error");
          stopPolling();
          toast({
            title:
              status === "denied"
                ? "Authorization denied"
                : "Authorization expired",
            description:
              status === "denied"
                ? "The authorization request was denied."
                : "The authorization request expired. Please try again.",
            variant: "destructive",
          });
          return;
        }

        // pending or slow_down — schedule next poll
        pollingRef.current = setTimeout(
          () => poll(token, runId),
          intervalRef.current * 1000,
        );
      } catch (error) {
        if (isUnmountedRef.current || runId !== runIdRef.current) return;

        // The device code is still valid at the provider for the rest of the
        // window, so a wifi switch or a proxy 502 should not force the user to
        // start over with a code they have already entered.
        consecutiveFailuresRef.current += 1;
        if (consecutiveFailuresRef.current < MAX_CONSECUTIVE_POLL_FAILURES) {
          pollingRef.current = setTimeout(
            () => poll(token, runId),
            intervalRef.current * 1000,
          );
          return;
        }

        setPhase("error");
        stopPolling();
        toast({
          title: "Device auth polling failed",
          description:
            error instanceof Error ? error.message : "Unexpected error",
          variant: "destructive",
        });
      }
    },
    [provider, onSuccess, queryClient, stopPolling],
  );

  async function connect() {
    // A second click would strand the first poll loop, which keeps polling a
    // token nothing reads any more. Bumping the run id also retires any poll
    // already in flight, which clearing the timeout cannot do.
    stopPolling();
    const runId = ++runIdRef.current;
    consecutiveFailuresRef.current = 0;
    setPhase("awaiting_user");
    try {
      const response = await postV1InitiateDeviceCodeOauthFlow(provider);

      if (isUnmountedRef.current || runId !== runIdRef.current) return;

      if (response.status !== 200) {
        throw new Error("Device auth initiation failed");
      }
      const data = response.data;
      setUserCode(data.user_code);
      setVerificationUrl(
        data.verification_url_complete || data.verification_url,
      );
      // Clamp: this drives a setTimeout, so a 0 or absurd value from the
      // provider would either spin the loop or stall it forever.
      intervalRef.current = Math.min(Math.max(data.interval || 5, 1), 60);

      // Start polling. Use the clamped value, not the raw one — an upstream
      // `interval: 0` would otherwise spin the first poll immediately.
      setPhase("polling");
      pollingRef.current = setTimeout(
        () => poll(data.state_token, runId),
        intervalRef.current * 1000,
      );
    } catch (error) {
      if (isUnmountedRef.current || runId !== runIdRef.current) return;
      setPhase("error");
      toast({
        title: "Device auth initiation failed",
        description:
          error instanceof Error ? error.message : "Unexpected error",
        variant: "destructive",
      });
    }
  }

  function cancel() {
    // Retires any in-flight poll as well as the scheduled one.
    runIdRef.current += 1;
    stopPolling();
    setPhase("idle");
    setUserCode("");
    setVerificationUrl("");
  }

  return {
    connect,
    cancel,
    phase,
    userCode,
    verificationUrl,
  };
}
