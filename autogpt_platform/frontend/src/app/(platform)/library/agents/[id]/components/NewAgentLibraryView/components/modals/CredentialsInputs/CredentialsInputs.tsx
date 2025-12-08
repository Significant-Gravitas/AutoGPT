import { useDeleteV1DeleteCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { IconKey } from "@/components/__legacy__/ui/icons";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import { DotsThreeVertical } from "@phosphor-icons/react";
import { NotionLogoIcon } from "@radix-ui/react-icons";
import { useQueryClient } from "@tanstack/react-query";
import { useContext, useEffect, useMemo, useState } from "react";
import {
  FaDiscord,
  FaGithub,
  FaGoogle,
  FaHubspot,
  FaKey,
  FaMedium,
  FaTwitter,
} from "react-icons/fa";
import { APIKeyCredentialsModal } from "./components/APIKeyCredentialsModal/APIKeyCredentialsModal";
import { HostScopedCredentialsModal } from "./components/HotScopedCredentialsModal/HotScopedCredentialsModal";
import { OAuthFlowWaitingModal } from "./components/OAuthWaitingModal/OAuthWaitingModal";
import { PasswordCredentialsModal } from "./components/PasswordCredentialsModal/PasswordCredentialsModal";

const fallbackIcon = FaKey;

// --8<-- [start:ProviderIconsEmbed]
// Provider icons mapping - uses fallback for unknown providers
export const providerIcons: Partial<
  Record<string, React.FC<{ className?: string }>>
> = {
  aiml_api: fallbackIcon,
  anthropic: fallbackIcon,
  apollo: fallbackIcon,
  e2b: fallbackIcon,
  github: FaGithub,
  google: FaGoogle,
  groq: fallbackIcon,
  http: fallbackIcon,
  notion: NotionLogoIcon,
  nvidia: fallbackIcon,
  discord: FaDiscord,
  d_id: fallbackIcon,
  google_maps: FaGoogle,
  jina: fallbackIcon,
  ideogram: fallbackIcon,
  linear: fallbackIcon,
  medium: FaMedium,
  mem0: fallbackIcon,
  ollama: fallbackIcon,
  openai: fallbackIcon,
  openweathermap: fallbackIcon,
  open_router: fallbackIcon,
  llama_api: fallbackIcon,
  pinecone: fallbackIcon,
  enrichlayer: fallbackIcon,
  slant3d: fallbackIcon,
  screenshotone: fallbackIcon,
  smtp: fallbackIcon,
  replicate: fallbackIcon,
  reddit: fallbackIcon,
  fal: fallbackIcon,
  revid: fallbackIcon,
  twitter: FaTwitter,
  unreal_speech: fallbackIcon,
  exa: fallbackIcon,
  hubspot: FaHubspot,
  smartlead: fallbackIcon,
  todoist: fallbackIcon,
  zerobounce: fallbackIcon,
};
// --8<-- [end:ProviderIconsEmbed]

export type OAuthPopupResultMessage = { message_type: "oauth_popup_result" } & (
  | {
      success: true;
      code: string;
      state: string;
    }
  | {
      success: false;
      message: string;
    }
);

type Props = {
  schema: BlockIOCredentialsSubSchema;
  className?: string;
  selectedCredentials?: CredentialsMetaInput;
  onSelectCredentials: (newValue?: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
  hideIfSingleCredentialAvailable?: boolean;
  onLoaded?: (loaded: boolean) => void;
};

export function CredentialsInput({
  schema,
  className,
  selectedCredentials,
  onSelectCredentials,
  siblingInputs,
  hideIfSingleCredentialAvailable = true,
  onLoaded,
}: Props) {
  const [isAPICredentialsModalOpen, setAPICredentialsModalOpen] =
    useState(false);
  const [
    isUserPasswordCredentialsModalOpen,
    setUserPasswordCredentialsModalOpen,
  ] = useState(false);
  const [isHostScopedCredentialsModalOpen, setHostScopedCredentialsModalOpen] =
    useState(false);
  const [isOAuth2FlowInProgress, setOAuth2FlowInProgress] = useState(false);
  const [oAuthPopupController, setOAuthPopupController] =
    useState<AbortController | null>(null);
  const [oAuthError, setOAuthError] = useState<string | null>(null);
  const [credentialToDelete, setCredentialToDelete] = useState<{
    id: string;
    title: string;
  } | null>(null);

  const api = useBackendAPI();
  const queryClient = useQueryClient();
  const credentials = useCredentials(schema, siblingInputs);
  const allProviders = useContext(CredentialsProvidersContext);

  const deleteCredentialsMutation = useDeleteV1DeleteCredentials({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: ["/api/integrations/credentials"],
        });
        queryClient.invalidateQueries({
          queryKey: [`/api/integrations/${credentials?.provider}/credentials`],
        });
        setCredentialToDelete(null);
        if (selectedCredentials?.id === credentialToDelete?.id) {
          onSelectCredentials(undefined);
        }
      },
    },
  });

  // Get the raw provider data to access ALL saved credentials (not filtered)
  const rawProvider = credentials
    ? allProviders?.[credentials.provider as keyof typeof allProviders]
    : null;

  // Report loaded state to parent
  useEffect(() => {
    if (onLoaded) {
      onLoaded(Boolean(credentials && credentials.isLoading === false));
    }
  }, [credentials, onLoaded]);

  // Deselect credentials if they do not exist (e.g. provider was changed)
  useEffect(() => {
    if (!credentials || !("savedCredentials" in credentials)) return;
    if (
      selectedCredentials &&
      !credentials.savedCredentials.some((c) => c.id === selectedCredentials.id)
    ) {
      onSelectCredentials(undefined);
    }
  }, [credentials, selectedCredentials, onSelectCredentials]);

  const { singleCredential } = useMemo(() => {
    if (!credentials || !("savedCredentials" in credentials)) {
      return {
        singleCredential: null,
      };
    }

    // Auto-select single credential if only one exists
    const single =
      credentials.savedCredentials.length === 1
        ? credentials.savedCredentials[0]
        : null;

    return {
      singleCredential: single,
    };
  }, [credentials]);

  // If only 1 credential is available, auto-select it
  useEffect(() => {
    if (
      singleCredential &&
      !selectedCredentials &&
      hideIfSingleCredentialAvailable
    ) {
      onSelectCredentials(singleCredential);
    }
  }, [
    singleCredential,
    selectedCredentials,
    onSelectCredentials,
    hideIfSingleCredentialAvailable,
  ]);

  if (!credentials || credentials.isLoading) {
    return null;
  }

  // Type guard to ensure we have the loaded credentials data
  if (!("savedCredentials" in credentials)) {
    return null;
  }

  const {
    provider,
    providerName,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    supportsHostScoped,
    savedCredentials,
    oAuthCallback,
  } = credentials;

  async function handleOAuthLogin() {
    setOAuthError(null);
    const { login_url, state_token } = await api.oAuthLogin(
      provider,
      schema.credentials_scopes,
    );
    setOAuth2FlowInProgress(true);
    const popup = window.open(login_url, "_blank", "popup=true");

    if (!popup) {
      throw new Error(
        "Failed to open popup window. Please allow popups for this site.",
      );
    }

    const controller = new AbortController();
    setOAuthPopupController(controller);
    controller.signal.onabort = () => {
      console.debug("OAuth flow aborted");
      setOAuth2FlowInProgress(false);
      popup.close();
    };

    const handleMessage = async (e: MessageEvent<OAuthPopupResultMessage>) => {
      console.debug("Message received:", e.data);
      if (
        typeof e.data != "object" ||
        !("message_type" in e.data) ||
        e.data.message_type !== "oauth_popup_result"
      ) {
        console.debug("Ignoring irrelevant message");
        return;
      }

      if (!e.data.success) {
        console.error("OAuth flow failed:", e.data.message);
        setOAuthError(`OAuth flow failed: ${e.data.message}`);
        setOAuth2FlowInProgress(false);
        return;
      }

      if (e.data.state !== state_token) {
        console.error("Invalid state token received");
        setOAuthError("Invalid state token received");
        setOAuth2FlowInProgress(false);
        return;
      }

      try {
        console.debug("Processing OAuth callback");
        const credentials = await oAuthCallback(e.data.code, e.data.state);
        console.debug("OAuth callback processed successfully");
        onSelectCredentials({
          id: credentials.id,
          type: "oauth2",
          title: credentials.title,
          provider,
        });
      } catch (error) {
        console.error("Error in OAuth callback:", error);
        setOAuthError(
          // type of error is unkown so we need to use String(error)
          `Error in OAuth callback: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      } finally {
        console.debug("Finalizing OAuth flow");
        setOAuth2FlowInProgress(false);
        controller.abort("success");
      }
    };

    console.debug("Adding message event listener");
    window.addEventListener("message", handleMessage, {
      signal: controller.signal,
    });

    setTimeout(
      () => {
        console.debug("OAuth flow timed out");
        controller.abort("timeout");
        setOAuth2FlowInProgress(false);
        setOAuthError("OAuth flow timed out");
      },
      5 * 60 * 1000,
    );
  }

  const ProviderIcon = providerIcons[provider] || fallbackIcon;
  const modals = (
    <>
      {supportsApiKey && (
        <APIKeyCredentialsModal
          schema={schema}
          open={isAPICredentialsModalOpen}
          onClose={() => setAPICredentialsModalOpen(false)}
          onCredentialsCreate={(credsMeta) => {
            onSelectCredentials(credsMeta);
            setAPICredentialsModalOpen(false);
          }}
          siblingInputs={siblingInputs}
        />
      )}
      {supportsOAuth2 && (
        <OAuthFlowWaitingModal
          open={isOAuth2FlowInProgress}
          onClose={() => oAuthPopupController?.abort("canceled")}
          providerName={providerName}
        />
      )}
      {supportsUserPassword && (
        <PasswordCredentialsModal
          schema={schema}
          open={isUserPasswordCredentialsModalOpen}
          onClose={() => setUserPasswordCredentialsModalOpen(false)}
          onCredentialsCreate={(creds) => {
            onSelectCredentials(creds);
            setUserPasswordCredentialsModalOpen(false);
          }}
          siblingInputs={siblingInputs}
        />
      )}
      {supportsHostScoped && (
        <HostScopedCredentialsModal
          schema={schema}
          open={isHostScopedCredentialsModalOpen}
          onClose={() => setHostScopedCredentialsModalOpen(false)}
          onCredentialsCreate={(creds) => {
            onSelectCredentials(creds);
            setHostScopedCredentialsModalOpen(false);
          }}
          siblingInputs={siblingInputs}
        />
      )}
    </>
  );

  function getActionButtonText(): string {
    if (supportsOAuth2) return "Connect a different account";
    if (supportsApiKey) return "Use a different API key";
    if (supportsUserPassword) return "Use a different username and password";
    if (supportsHostScoped) return "Use different headers";
    return "Add credentials";
  }

  function handleActionButtonClick() {
    if (supportsOAuth2) {
      handleOAuthLogin();
    } else if (supportsApiKey) {
      setAPICredentialsModalOpen(true);
    } else if (supportsUserPassword) {
      setUserPasswordCredentialsModalOpen(true);
    } else if (supportsHostScoped) {
      setHostScopedCredentialsModalOpen(true);
    }
  }

  function handleCredentialSelect(credentialId: string) {
    const selectedCreds =
      savedCredentials.find((c) => c.id === credentialId) ||
      (selectedCredentials?.id === credentialId ? selectedCredentials : null);
    if (selectedCreds) {
      onSelectCredentials({
        id: selectedCreds.id,
        type: selectedCreds.type,
        provider: provider,
        title: (selectedCreds as any).title,
      });
    }
  }

  const displayName = toDisplayName(provider);

  // Use raw provider credentials (all saved credentials) instead of filtered ones
  // The filtering in useCredentials is for compatibility checking, but we want to show all credentials
  const allSavedCredentials = rawProvider?.savedCredentials || savedCredentials;

  // Combine saved credentials with selected credential if it's not already in the list
  const credentialsToShow = (() => {
    const creds = [...allSavedCredentials];
    if (
      selectedCredentials &&
      !creds.some((c) => c.id === selectedCredentials.id)
    ) {
      // If selected credential is not in saved list, add it
      creds.push({
        id: selectedCredentials.id,
        type: selectedCredentials.type,
        title: selectedCredentials.title || "Selected credential",
        provider: provider,
      } as any);
    }
    return creds;
  })();

  const hasCredentialsToShow = credentialsToShow.length > 0;

  return (
    <div className={cn("mb-6", className)}>
      <div className="mb-2 flex items-center gap-2">
        <Text variant="large-medium">{displayName} credentials</Text>
        <InformationTooltip description={schema.description} />
      </div>

      {hasCredentialsToShow ? (
        <>
          <div className="mb-4 space-y-2">
            {credentialsToShow.map((credential) => {
              const isSelected = selectedCredentials?.id === credential.id;
              return (
                <div
                  key={credential.id}
                  className={cn(
                    "flex items-center gap-3 rounded-medium border p-3 transition-colors",
                    isSelected
                      ? "border-purple-500 bg-white"
                      : "border-zinc-200 bg-white hover:border-gray-300",
                  )}
                  onClick={() => handleCredentialSelect(credential.id)}
                >
                  <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gray-900">
                    <ProviderIcon className="h-3 w-3 text-white" />
                  </div>
                  <IconKey className="h-5 w-5 shrink-0 text-zinc-800" />
                  <div className="flex min-w-0 flex-1 flex-nowrap items-center gap-4">
                    <Text variant="body" className="tracking-tight">
                      {credential.title ||
                        credential.username ||
                        `Your ${displayName} account`}
                    </Text>
                    <Text
                      variant="large"
                      className="relative top-1 font-mono tracking-tight"
                    >
                      {"*".repeat(30)}
                    </Text>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        className="ml-auto shrink-0 rounded p-1 hover:bg-gray-100"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <DotsThreeVertical className="h-5 w-5 text-gray-400" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={(e) => {
                          e.stopPropagation();
                          setCredentialToDelete({
                            id: credential.id,
                            title:
                              credential.title ||
                              credential.username ||
                              `Your ${displayName} account`,
                          });
                        }}
                      >
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              );
            })}
          </div>
          <Button
            variant="secondary"
            size="small"
            onClick={handleActionButtonClick}
            className="w-fit"
          >
            {getActionButtonText()}
          </Button>
        </>
      ) : (
        <Button
          variant="secondary"
          size="small"
          onClick={handleActionButtonClick}
          className="w-fit"
        >
          {getActionButtonText()}
        </Button>
      )}

      {modals}
      {oAuthError && (
        <div className="mt-2 text-red-500">Error: {oAuthError}</div>
      )}

      <Dialog
        controlled={{
          isOpen: credentialToDelete !== null,
          set: (open) => {
            if (!open) setCredentialToDelete(null);
          },
        }}
        title="Delete credential"
        styling={{ maxWidth: "32rem" }}
      >
        <Dialog.Content>
          <Text variant="large">
            Are you sure you want to delete &quot;{credentialToDelete?.title}
            &quot;? This action cannot be undone.
          </Text>
          <Dialog.Footer>
            <Button
              variant="secondary"
              onClick={() => setCredentialToDelete(null)}
              disabled={deleteCredentialsMutation.isPending}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                if (credentialToDelete && credentials) {
                  deleteCredentialsMutation.mutate({
                    provider: credentials.provider,
                    credId: credentialToDelete.id,
                  });
                }
              }}
              loading={deleteCredentialsMutation.isPending}
            >
              Delete
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
