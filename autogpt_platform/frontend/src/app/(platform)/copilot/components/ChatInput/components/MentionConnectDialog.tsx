"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { ConnectServiceDialog } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/ConnectServiceDialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { grantToExpert } from "@/services/experts/grant-to-expert";
import { useQueryClient } from "@tanstack/react-query";
import type { ActiveMention } from "../useChatMentions";

interface Props {
  /** The `@query` "Connect an integration" was chosen from; null keeps the
   *  dialog closed. */
  request: ActiveMention | null;
  /** Expert the chat is scoped to. Its @ list only offers accounts it has
   *  been granted, so a new connection is granted to it. */
  expertId?: string | null;
  expertName?: string | null;
  onCancel: () => void;
  onConnected: (request: ActiveMention, credentialId: string) => void;
}

/** The composer's own copy of the "Connect a service" dialog, opened from
 *  the @ mention list and searched for what was typed after the `@`. */
export function MentionConnectDialog({
  request,
  expertId,
  expertName,
  onCancel,
  onConnected,
}: Props) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  async function handleConnected(credential: CredentialsMetaResponse) {
    if (!request) return;
    if (expertId) {
      const ok = await grantToExpert(queryClient, expertId, credential.id);
      if (!ok) {
        toast({
          title: `Connected, but could not give ${expertName ?? "this expert"} access`,
          description: "Grant it from the Integrations tab.",
          variant: "destructive",
        });
      }
    }
    onConnected(request, credential.id);
  }

  return (
    // The composer renders this inside its form, and React carries a submit
    // from the dialog's API key form up through the portal to it, which
    // would send the half-written message.
    <div className="contents" onSubmit={(e) => e.stopPropagation()}>
      <ConnectServiceDialog
        preferMcp
        open={request !== null}
        initialQuery={request?.query.trim()}
        onOpenChange={(open) => {
          if (!open) onCancel();
        }}
        onConnected={handleConnected}
      />
    </div>
  );
}
