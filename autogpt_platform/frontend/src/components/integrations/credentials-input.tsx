import SchemaTooltip from "@/components/SchemaTooltip";
import { Button } from "@/components/ui/button";
import { IconKey, IconKeyPlus, IconUserPlus } from "@/components/ui/icons";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { getHostFromUrl } from "@/lib/utils/url";
import { NotionLogoIcon } from "@radix-ui/react-icons";
import { FC, useEffect, useMemo, useState } from "react";
import {
  FaDiscord,
  FaGithub,
  FaGoogle,
  FaHubspot,
  FaKey,
  FaMedium,
  FaTwitter,
} from "react-icons/fa";
import { APIKeyCredentialsModal } from "./api-key-credentials-modal";
import { HostScopedCredentialsModal } from "./host-scoped-credentials-modal";
import { OAuth2FlowWaitingModal } from "./oauth2-flow-waiting-modal";
import { UserPasswordCredentialsModal } from "./user-password-credentials-modal";

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

export const CredentialsInput: FC<{
  schema: BlockIOCredentialsSubSchema;
  className?: string;
  selectedCredentials?: CredentialsMetaInput;
  onSelectCredentials: (newValue?: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
  hideIfSingleCredentialAvailable?: boolean;
}> = ({
  schema,
  className,
  selectedCredentials,
  onSelectCredentials,
  siblingInputs,
  hideIfSingleCredentialAvailable = true,
}) => {
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

  const api = useBackendAPI();
  const credentials = useCredentials(schema, siblingInputs);

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

  const { hasRelevantCredentials, singleCredential } = useMemo(() => {
    if (!credentials || !("savedCredentials" in credentials)) {
      return {
        hasRelevantCredentials: false,
        singleCredential: null,
      };
    }

    // Simple logic: if we have any saved credentials, we have relevant credentials
    const hasRelevant = credentials.savedCredentials.length > 0;

    // Auto-select single credential if only one exists
    const single =
      credentials.savedCredentials.length === 1
        ? credentials.savedCredentials[0]
        : null;

    return {
      hasRelevantCredentials: hasRelevant,
      singleCredential: single,
    };
  }, [credentials]);

  // If only 1 credential is available, auto-select it and hide this input
  useEffect(() => {
    if (singleCredential && !selectedCredentials) {
      onSelectCredentials(singleCredential);
    }
  }, [singleCredential, selectedCredentials, onSelectCredentials]);

  if (
    !credentials ||
    credentials.isLoading ||
    (singleCredential && hideIfSingleCredentialAvailable)
  ) {
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
        <OAuth2FlowWaitingModal
          open={isOAuth2FlowInProgress}
          onClose={() => oAuthPopupController?.abort("canceled")}
          providerName={providerName}
        />
      )}
      {supportsUserPassword && (
        <UserPasswordCredentialsModal
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

  const fieldHeader = (
    <div className="mb-2 flex gap-1">
      <span className="text-m green text-gray-900">
        {providerName} Credentials
      </span>
      <SchemaTooltip description={schema.description} />
    </div>
  );

  // Show credentials creation UI when no relevant credentials exist
  if (!hasRelevantCredentials) {
    return (
      <div>
        {fieldHeader}

        <div className={cn("flex flex-row space-x-2", className)}>
          {supportsOAuth2 && (
            <Button onClick={handleOAuthLogin}>
              <ProviderIcon className="mr-2 h-4 w-4" />
              {"Sign in with " + providerName}
            </Button>
          )}
          {supportsApiKey && (
            <Button onClick={() => setAPICredentialsModalOpen(true)}>
              <ProviderIcon className="mr-2 h-4 w-4" />
              Enter API key
            </Button>
          )}
          {supportsUserPassword && (
            <Button onClick={() => setUserPasswordCredentialsModalOpen(true)}>
              <ProviderIcon className="mr-2 h-4 w-4" />
              Enter username and password
            </Button>
          )}
          {supportsHostScoped && credentials.discriminatorValue && (
            <Button onClick={() => setHostScopedCredentialsModalOpen(true)}>
              <ProviderIcon className="mr-2 h-4 w-4" />
              {`Enter sensitive headers for ${getHostFromUrl(credentials.discriminatorValue)}`}
            </Button>
          )}
        </div>
        {modals}
        {oAuthError && (
          <div className="mt-2 text-red-500">Error: {oAuthError}</div>
        )}
      </div>
    );
  }

  function handleValueChange(newValue: string) {
    if (newValue === "sign-in") {
      // Trigger OAuth2 sign in flow
      handleOAuthLogin();
    } else if (newValue === "add-api-key") {
      // Open API key dialog
      setAPICredentialsModalOpen(true);
    } else if (newValue === "add-user-password") {
      // Open user password dialog
      setUserPasswordCredentialsModalOpen(true);
    } else if (newValue === "add-host-scoped") {
      // Open host-scoped credentials dialog
      setHostScopedCredentialsModalOpen(true);
    } else {
      const selectedCreds = savedCredentials.find((c) => c.id == newValue)!;

      onSelectCredentials({
        id: selectedCreds.id,
        type: selectedCreds.type,
        provider: provider,
        // title: customTitle, // TODO: add input for title
      });
    }
  }

  // Saved credentials exist
  return (
    <div>
      {fieldHeader}

      <Select value={selectedCredentials?.id} onValueChange={handleValueChange}>
        <SelectTrigger>
          <SelectValue placeholder={schema.placeholder} />
        </SelectTrigger>
        <SelectContent className="nodrag">
          {savedCredentials
            .filter((c) => c.type == "oauth2")
            .map((credentials, index) => (
              <SelectItem key={index} value={credentials.id}>
                <ProviderIcon className="mr-2 inline h-4 w-4" />
                {credentials.title ||
                  credentials.username ||
                  `Your ${providerName} account`}
              </SelectItem>
            ))}
          {savedCredentials
            .filter((c) => c.type == "api_key")
            .map((credentials, index) => (
              <SelectItem key={index} value={credentials.id}>
                <ProviderIcon className="mr-2 inline h-4 w-4" />
                <IconKey className="mr-1.5 inline" />
                {credentials.title}
              </SelectItem>
            ))}
          {savedCredentials
            .filter((c) => c.type == "user_password")
            .map((credentials, index) => (
              <SelectItem key={index} value={credentials.id}>
                <ProviderIcon className="mr-2 inline h-4 w-4" />
                <IconUserPlus className="mr-1.5 inline" />
                {credentials.title}
              </SelectItem>
            ))}
          {savedCredentials
            .filter((c) => c.type == "host_scoped")
            .map((credentials, index) => (
              <SelectItem key={index} value={credentials.id}>
                <ProviderIcon className="mr-2 inline h-4 w-4" />
                <IconKey className="mr-1.5 inline" />
                {credentials.title}
              </SelectItem>
            ))}
          <SelectSeparator />
          {supportsOAuth2 && (
            <SelectItem value="sign-in">
              <IconUserPlus className="mr-1.5 inline" />
              Sign in with {providerName}
            </SelectItem>
          )}
          {supportsApiKey && (
            <SelectItem value="add-api-key">
              <IconKeyPlus className="mr-1.5 inline" />
              Add new API key
            </SelectItem>
          )}
          {supportsUserPassword && (
            <SelectItem value="add-user-password">
              <IconUserPlus className="mr-1.5 inline" />
              Add new user password
            </SelectItem>
          )}
          {supportsHostScoped && (
            <SelectItem value="add-host-scoped">
              <IconKey className="mr-1.5 inline" />
              Add host-scoped headers
            </SelectItem>
          )}
        </SelectContent>
      </Select>
      {modals}
      {oAuthError && (
        <div className="mt-2 text-red-500">Error: {oAuthError}</div>
      )}
    </div>
  );
};
