"use client";

import { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { useQueryClient } from "@tanstack/react-query";

import {
  getGetV1ListCredentialsQueryKey,
  postV1CreateCredentials,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { toast } from "@/components/molecules/Toast/use-toast";

import {
  apiKeyConnectSchema,
  type ApiKeyConnectFormValues,
} from "./schema";

interface Args {
  provider: string;
  onSuccess: () => void;
}

function toUnixSeconds(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) return undefined;
  return Math.floor(ms / 1000);
}

export function useApiKeyConnectForm({ provider, onSuccess }: Args) {
  const queryClient = useQueryClient();
  const [isPending, setIsPending] = useState(false);

  const form = useForm<ApiKeyConnectFormValues>({
    resolver: zodResolver(apiKeyConnectSchema),
    defaultValues: { title: "", apiKey: "", expiresAt: "" },
    mode: "onChange",
  });

  async function handleSubmit(values: ApiKeyConnectFormValues) {
    setIsPending(true);
    try {
      const response = await postV1CreateCredentials(provider, {
        provider,
        type: "api_key",
        title: values.title,
        api_key: values.apiKey,
        expires_at: toUnixSeconds(values.expiresAt),
      });

      if (response.status !== 200) {
        throw new Error("Failed to save API key");
      }

      toast({ title: "API key saved", variant: "success" });
      await queryClient.invalidateQueries({
        queryKey: getGetV1ListCredentialsQueryKey(),
      });
      onSuccess();
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
