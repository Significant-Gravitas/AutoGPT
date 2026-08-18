"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { getGetV1ListCredentialsQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { customMutator } from "@/app/api/mutators/custom-mutator";
import { toast } from "@/components/molecules/Toast/use-toast";

interface DeviceAuthInitiateResponse {
  state_token: string;
  device_code: string;
  user_code: string;
  verification_url: string;
  verification_url_complete: string | null;
  expires_in: number;
  interval: number;
}

interface DeviceAuthPollResponse {
  status: "pending" | "slow_down" | "approved" | "denied" | "expired";
  credentials: unknown | null;
}

interface Args {
  provider: string;
  onSuccess: () => void;
}

type Phase = "idle" | "awaiting_user" | "polling" | "done" | "error";

export function useDeviceAuthConnect({ provider, onSuccess }: Args) {
  const queryClient = useQueryClient();
  const [phase, setPhase] = useState<Phase>("idle");
  const [userCode, setUserCode] = useState("");
  const [verificationUrl, setVerificationUrl] = useState("");
  const [stateToken, setStateToken] = useState("");

  const isUnmountedRef = useRef(false);
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intervalRef = useRef(5);

  useEffect(() => {
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
        const response = await customMutator<{
          data: DeviceAuthPollResponse;
          status: number;
          headers: Headers;
        }>(`/integrations/${provider}/device-auth/poll`, {
          method: "POST",
          body: JSON.stringify({ state_token: token }),
          headers: { "Content-Type": "application/json" },
        });

        if (isUnmountedRef.current) return;

        const { status } = response.data;

        if (status === "approved") {
          setPhase("done");
          stopPolling();
          toast({ title: "Connected via device auth", variant: "success" });
          await queryClient.invalidateQueries({
            queryKey: getGetV1ListCredentialsQueryKey(),
          });
          onSuccess();
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
    setPhase("awaiting_user");
    try {
      const response = await customMutator<{
        data: DeviceAuthInitiateResponse;
        status: number;
        headers: Headers;
      }>(`/integrations/${provider}/device-auth/initiate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (isUnmountedRef.current) return;

      const data = response.data;
      setUserCode(data.user_code);
      setVerificationUrl(
        data.verification_url_complete || data.verification_url,
      );
      setStateToken(data.state_token);
      intervalRef.current = data.interval;

      // Start polling
      setPhase("polling");
      pollingRef.current = setTimeout(
        () => poll(data.state_token),
        data.interval * 1000,
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
