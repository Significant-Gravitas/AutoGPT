"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV1ListCredentialsQueryKey,
  getV1InitiateOauthFlow,
  postV1ExchangeOauthCodeForTokens,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { toast } from "@/components/molecules/Toast/use-toast";
import { openOAuthPopup } from "@/lib/oauth-popup";

interface Args {
  provider: string;
  onSuccess: () => void;
}

export function useOAuthConnect({ provider, onSuccess }: Args) {
  const queryClient = useQueryClient();
  const [isPending, setIsPending] = useState(false);
  const abortRef = useRef<(() => void) | null>(null);
  const isUnmountedRef = useRef(false);

  useEffect(() => {
    return () => {
      isUnmountedRef.current = true;
      abortRef.current?.();
    };
  }, []);

  async function connect() {
    setIsPending(true);
    try {
      const initiateResponse = await getV1InitiateOauthFlow(provider);
      // customMutator rejects non-2xx, so this branch is unreachable at
      // runtime — it exists only to narrow the discriminated union so the
      // 200-only LoginResponse shape is accessible below.
      if (initiateResponse.status !== 200) {
        throw new Error("Unexpected OAuth initiate response");
      }
      const { login_url, state_token } = initiateResponse.data;

      const { promise, cleanup } = openOAuthPopup(login_url, {
        stateToken: state_token,
      });
      abortRef.current = () => cleanup.abort("unmounted");

      const { code, state } = await promise;
      abortRef.current = null;

      await postV1ExchangeOauthCodeForTokens(provider, {
        code,
        state_token: state,
      });

      toast({ title: "Connected via OAuth", variant: "success" });
      await queryClient.invalidateQueries({
        queryKey: getGetV1ListCredentialsQueryKey(),
      });
      onSuccess();
    } catch (error) {
      if (isUnmountedRef.current) return;
      toast({
        title: "OAuth connection failed",
        description:
          error instanceof Error ? error.message : "Unexpected error",
        variant: "destructive",
      });
    } finally {
      setIsPending(false);
    }
  }

  return { connect, isPending };
}
