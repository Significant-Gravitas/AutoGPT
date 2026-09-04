"use client";

import { useContext, useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { useQueryClient } from "@tanstack/react-query";

import { postV1CreateCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { invalidateConnectionQueries } from "@/lib/react-query/invalidateConnections";
import { CredentialsActionsContext } from "@/providers/agent-credentials/credentials-provider";

import { apiKeyConnectSchema, type ApiKeyConnectFormValues } from "./schema";

interface Args {
  provider: string;
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

function toUnixSeconds(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) return undefined;
  return Math.floor(ms / 1000);
}

export function useApiKeyConnectForm({ provider, onSuccess }: Args) {
  const queryClient = useQueryClient();
  const credentialsActions = useContext(CredentialsActionsContext);
  const [isPending, setIsPending] = useState(false);

  const form = useForm<ApiKeyConnectFormValues>({
    resolver: zodResolver(apiKeyConnectSchema),
    defaultValues: { title: "", apiKey: "", expiresAt: "" },
    mode: "onChange",
  });

  async function handleSubmit(values: ApiKeyConnectFormValues) {
    setIsPending(true);
    try {
      // customMutator throws on non-2xx, so reaching this line means success.
      // Trust HTTP semantics rather than pinning to a specific 2xx code —
      // proxies / future backend changes can swap 201 ↔ 200 without this
      // breaking and silently failing in production.
      const created = await postV1CreateCredentials(provider, {
        provider,
        type: "api_key",
        title: values.title,
        api_key: values.apiKey,
        expires_at: toUnixSeconds(values.expiresAt),
      });

      toast({ title: "API key saved", variant: "success" });
      await invalidateConnectionQueries(queryClient);
      // Same reason as the OAuth branch: invalidation emits no cache event
      // unless something already subscribed to the credentials query.
      credentialsActions?.reload();
      // Narrow by shape rather than by status code: the comment above keeps
      // this flow indifferent to a 201 ↔ 200 swap, and only the success
      // payload carries an id.
      onSuccess("id" in created.data ? created.data : undefined);
    } catch (error) {
      toast({
        title: "Couldn't save API key",
        description:
          error instanceof Error ? error.message : "Unexpected error",
        variant: "destructive",
      });
    } finally {
      setIsPending(false);
    }
  }

  return { form, handleSubmit, isPending };
}
