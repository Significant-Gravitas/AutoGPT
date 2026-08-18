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

export function useDeviceAuthConnect({ provider, onSuccess }: Args) {
  const queryClient = useQueryClient();
  const [phase, setPhase] = useState<Phase>("idle");
  const [userCode, setUserCode] = useState("");
  const [verificationUrl, setVerificationUrl] = useState("");
  const [, setStateToken] = useState("");

  const isUnmountedRef = useRef(false);
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intervalRef = useRef(5);

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
    async (token: string) => {
      if (isUnmountedRef.current) return;

      try {
        const response = await postV1PollDeviceCodeOauthFlowForCompletion(
          provider,
          { state_token: token },
        );

        if (isUnmountedRef.current) return;

        if (response.status !== 200) {
          throw new Error("Device auth poll failed");
        }
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
          () => poll(token),
          intervalRef.current * 1000,
        );
      } catch (error) {
        if (isUnmountedRef.current) return;
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
    // token nothing reads any more.
    stopPolling();
    setPhase("awaiting_user");
    try {
      const response = await postV1InitiateDeviceCodeOauthFlow(provider);

      if (isUnmountedRef.current) return;

      if (response.status !== 200) {
        throw new Error("Device auth initiation failed");
      }
      const data = response.data;
      setUserCode(data.user_code);
      setVerificationUrl(
        data.verification_url_complete || data.verification_url,
      );
      setStateToken(data.state_token);
      // Clamp: this drives a setTimeout, so a 0 or absurd value from the
      // provider would either spin the loop or stall it forever.
      intervalRef.current = Math.min(Math.max(data.interval || 5, 1), 60);

      // Start polling. Use the clamped value, not the raw one — an upstream
      // `interval: 0` would otherwise spin the first poll immediately.
      setPhase("polling");
      pollingRef.current = setTimeout(
        () => poll(data.state_token),
        intervalRef.current * 1000,
      );
    } catch (error) {
      if (isUnmountedRef.current) return;
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
    stopPolling();
    setPhase("idle");
    setUserCode("");
    setVerificationUrl("");
    setStateToken("");
  }

  return {
    connect,
    cancel,
    phase,
    userCode,
    verificationUrl,
    isPending: phase === "awaiting_user" || phase === "polling",
  };
}
