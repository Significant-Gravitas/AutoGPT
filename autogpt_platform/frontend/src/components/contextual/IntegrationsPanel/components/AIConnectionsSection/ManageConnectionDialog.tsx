"use client";

import { useState } from "react";
import { useGetV1CodexAccount } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { useOAuthConnect } from "../ConnectServiceDialog/components/DetailView/useOAuthConnect";
import { DeleteConfirmDialog } from "../DeleteConfirmDialog/DeleteConfirmDialog";
import { useDeleteIntegration } from "../hooks/useDeleteIntegration";
import { microsoftAccessLabels } from "./helpers";

interface Props {
  connection: AIConnectionOffer | null;
  credential?: CredentialsMetaResponse;
  onOpenChange: (open: boolean) => void;
}

/**
 * Single-account first pass. Deliberately shows no usage figure and no reset
 * window: the only numbers available would be from whenever they were last
 * observed, and a stale window presented as current is worse than none —
 * especially on the reconnect path, where authorization is already invalid.
 */
export function ManageConnectionDialog({
  connection,
  credential,
  onOpenChange,
}: Props) {
  if (connection?.auth_provider === "microsoft_365_copilot") {
    return (
      <Microsoft365CopilotConnectionDialog
        connection={connection}
        credential={credential}
        onOpenChange={onOpenChange}
      />
    );
  }

  return (
    <ChatGPTConnectionDialog
      connection={connection}
      credential={credential}
      onOpenChange={onOpenChange}
    />
  );
}

function ChatGPTConnectionDialog({
  connection,
  credential,
  onOpenChange,
}: Props) {
  const credentialId = connection?.credential_id ?? null;
  const [forceMessage, setForceMessage] = useState<string | null>(null);
  const account = credential?.username ?? undefined;

  // Asked for only while this dialog is open. Reading it leases a runtime
  // against the provider, so it must never run on the list behind — and it
  // means what is shown was true a moment ago rather than whenever the
  // connection was last used.
  const accountQuery = useGetV1CodexAccount(credentialId ?? "", {
    query: {
      enabled: credentialId !== null,
      staleTime: 0,
      retry: false,
      // A window-focus refetch would lease another provider runtime while
      // the same snapshot is already visible. Reopening Manage is the user's
      // explicit refresh boundary.
      refetchOnWindowFocus: false,
    },
  });
  const accountResponse =
    accountQuery.data?.status === 200 ? accountQuery.data.data : undefined;
  const snapshot = accountResponse?.connected ? accountResponse : undefined;
  const requiresReconnect =
    accountResponse !== undefined && !accountResponse.connected;

  function setManageOpen(open: boolean) {
    if (!open) setForceMessage(null);
    onOpenChange(open);
  }

  const { connect, isPending: isReconnecting } = useOAuthConnect({
    provider: "codex",
    onSuccess: () => setManageOpen(false),
  });

  const { remove, isPending: isDisconnecting } = useDeleteIntegration();

  async function disconnect(force = false) {
    if (!credentialId) return;
    const target = {
      id: credentialId,
      provider: "codex",
      name: account ?? "ChatGPT",
    };
    const result = await remove([target], force);
    const confirmation = result.needsConfirmation[0];
    if (confirmation) {
      setForceMessage(confirmation.message);
      return;
    }
    if (result.succeeded.length === 0) return;

    setForceMessage(null);
    setManageOpen(false);
  }

  return (
    <>
      <Dialog
        title="ChatGPT"
        styling={{ maxWidth: "28rem" }}
        controlled={{
          isOpen: connection !== null && forceMessage === null,
          set: (open) => {
            // Opening the force-confirmation dialog intentionally hides this
            // one. Radix reports that controlled close through this callback;
            // keep the connection selected so Cancel can return here.
            if (!open && forceMessage !== null) return;
            setManageOpen(open);
          },
        }}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <Text variant="small" className="text-[#8A8A90]">
                Connected account
              </Text>
              <Text variant="body-medium" className="text-black">
                {snapshot?.email ?? account ?? "Your ChatGPT account"}
              </Text>
              {accountQuery.isLoading ? (
                <Text variant="small" className="text-[#8A8A90]">
                  Checking with ChatGPT…
                </Text>
              ) : accountQuery.isError ? (
                <Text variant="small" className="text-[#8A8A90]">
                  Could not reach ChatGPT to confirm the plan on this account.
                </Text>
              ) : requiresReconnect ? (
                <Text variant="small" className="text-amber-700">
                  ChatGPT says this connection needs to be reconnected.
                </Text>
              ) : (
                snapshot?.plan_type && (
                  <span className="mt-1 flex flex-wrap gap-2">
                    <span className="rounded-[10px] bg-[#F1EBFF] px-2 py-[2px] text-[13px] font-medium leading-[20px] text-[#4A25AD]">
                      {snapshot.plan_type} plan
                    </span>
                  </span>
                )
              )}
            </div>

            <Text variant="small" className="text-[#8A8A90]">
              ChatGPT identifies the workspace behind this connection only by an
              internal id, so it cannot be named here yet.
            </Text>

            <Text variant="small" className="text-[#505057]">
              {connection?.is_default
                ? "New chats start on this connection and run on your ChatGPT plan."
                : "Chats you route here run on your ChatGPT plan instead of AutoGPT credits."}
            </Text>

            <div className="flex flex-col gap-2 pt-2 sm:flex-row sm:justify-end">
              <Button
                variant="secondary"
                size="small"
                onClick={connect}
                loading={isReconnecting}
                disabled={isDisconnecting}
              >
                Reconnect
              </Button>
              <Button
                variant="destructive"
                size="small"
                onClick={() => void disconnect()}
                loading={isDisconnecting}
                disabled={isReconnecting}
              >
                Disconnect
              </Button>
            </div>

            <Text variant="small" className="text-[#8A8A90]">
              Disconnecting stops new chats running on your ChatGPT plan. Your
              chat history is kept, and your other integrations are unaffected.
            </Text>
          </div>
        </Dialog.Content>
      </Dialog>

      <DeleteConfirmDialog
        open={forceMessage !== null}
        onOpenChange={(open) => {
          if (!open) setForceMessage(null);
        }}
        itemNames={[account ?? "ChatGPT"]}
        isPending={isDisconnecting}
        onConfirm={() => void disconnect(true)}
        variant="force"
        notice={forceMessage ?? undefined}
      />
    </>
  );
}

interface MicrosoftProps {
  connection: AIConnectionOffer;
  credential?: CredentialsMetaResponse;
  onOpenChange: (open: boolean) => void;
}

function Microsoft365CopilotConnectionDialog({
  connection,
  credential,
  onOpenChange,
}: MicrosoftProps) {
  const { remove, isPending: isDisconnecting } = useDeleteIntegration();
  const credentialId = connection.credential_id;
  const accessLabels = microsoftAccessLabels(credential?.scopes);
  const displayName =
    credential?.title && credential.title !== "Microsoft 365 Copilot"
      ? credential.title
      : undefined;

  async function disconnect() {
    if (!credentialId) return;
    await remove([
      {
        id: credentialId,
        provider: "microsoft_365_copilot",
        name: credential?.username ?? "Microsoft 365 Copilot",
      },
    ]);
    onOpenChange(false);
  }

  return (
    <Dialog
      title="Microsoft 365 Copilot"
      styling={{ maxWidth: "32rem" }}
      controlled={{ isOpen: true, set: onOpenChange }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-5">
          <div className="flex flex-col gap-1">
            <Text variant="small" className="text-[#8A8A90]">
              Connected work account
            </Text>
            <Text variant="body-medium" className="text-black">
              {displayName ??
                credential?.username ??
                "Microsoft 365 Copilot account"}
            </Text>
            {displayName && credential?.username && (
              <Text variant="small" className="text-[#505057]">
                {credential.username}
              </Text>
            )}
            {!credential?.username && (
              <Text variant="small" className="text-[#8A8A90]">
                Reconnect once to show the signed-in account here.
              </Text>
            )}
          </div>

          <div className="flex flex-col gap-2">
            <Text variant="small-medium" className="text-[#505057]">
              Access granted
            </Text>
            {accessLabels.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {accessLabels.map((label) => (
                  <span
                    key={label}
                    className="rounded-[10px] bg-[#EFF1F4] px-2 py-[2px] text-[13px] font-medium leading-[20px] text-[#505057]"
                  >
                    {label}
                  </span>
                ))}
              </div>
            ) : (
              <Text variant="small" className="text-[#8A8A90]">
                Reconnect once to refresh the access shown here.
              </Text>
            )}
          </div>

          <Text variant="small" className="text-[#505057]">
            Microsoft still limits every answer to content this work account can
            already access in Microsoft 365.
          </Text>

          <div className="flex justify-end pt-1">
            <Button
              variant="destructive"
              size="small"
              onClick={disconnect}
              loading={isDisconnecting}
            >
              Disconnect
            </Button>
          </div>

          <Text variant="small" className="text-[#8A8A90]">
            Disconnecting stops new chats using this Copilot subscription. Your
            chat history is kept, and your other integrations are unaffected.
          </Text>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
