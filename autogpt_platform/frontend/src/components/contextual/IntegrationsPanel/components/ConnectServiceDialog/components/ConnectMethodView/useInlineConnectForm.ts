"use client";

import { useContext, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { postV1CreateCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { PostV1CreateCredentialsBody } from "@/app/api/__generated__/models/postV1CreateCredentialsBody";
import { toast } from "@/components/molecules/Toast/use-toast";
import { invalidateConnectionQueries } from "@/lib/react-query/invalidateConnections";
import { CredentialsActionsContext } from "@/providers/agent-credentials/credentials-provider";

interface Args {
  provider: string;
  successTitle: string;
  failureTitle: string;
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

/** Submit path for the method cards that save a credential themselves rather
 *  than through the panel footer's Continue — host-scoped and user/password. */
export function useInlineConnectForm({
  provider,
  successTitle,
  failureTitle,
  onSuccess,
}: Args) {
  const queryClient = useQueryClient();
  const credentialsActions = useContext(CredentialsActionsContext);
  const [isPending, setIsPending] = useState(false);

  async function submit(body: PostV1CreateCredentialsBody) {
    setIsPending(true);
    try {
      // customMutator throws on non-2xx, so reaching this line means success.
      const created = await postV1CreateCredentials(provider, body);

      toast({ title: successTitle, variant: "success" });
      await invalidateConnectionQueries(queryClient);
      // Invalidation emits no cache event unless something already
      // subscribed to the credentials query.
      credentialsActions?.reload();
      // Narrow by shape, not status code, so a 201 ↔ 200 swap cannot break
      // this; only the success payload carries an id.
      onSuccess("id" in created.data ? created.data : undefined);
    } catch (error) {
      toast({
        title: failureTitle,
        description:
          error instanceof Error ? error.message : "Unexpected error",
        variant: "destructive",
      });
    } finally {
      setIsPending(false);
    }
  }

  return { submit, isPending };
}
