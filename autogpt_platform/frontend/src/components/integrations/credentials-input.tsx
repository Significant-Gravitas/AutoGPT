import { z } from "zod";
import { cn } from "@/lib/utils";
import { useForm } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import useCredentials from "@/hooks/useCredentials";
import { zodResolver } from "@hookform/resolvers/zod";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { NotionLogoIcon } from "@radix-ui/react-icons";
import { FaGithub, FaGoogle } from "react-icons/fa";
import { FC, useMemo, useState } from "react";
import {
  APIKeyCredentials,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import {
  IconKey,
  IconKeyPlus,
  IconUser,
  IconUserPlus,
} from "@/components/ui/icons";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// --8<-- [start:ProviderIconsEmbed]
const providerIcons: Record<string, React.FC<{ className?: string }>> = {
  github: FaGithub,
  google: FaGoogle,
  notion: NotionLogoIcon,
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
  className?: string;
  selectedCredentials?: CredentialsMetaInput;
  onSelectCredentials: (newValue: CredentialsMetaInput) => void;
}> = ({ className, selectedCredentials, onSelectCredentials }) => {
  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const credentials = useCredentials();
  const [isAPICredentialsModalOpen, setAPICredentialsModalOpen] =
    useState(false);
  const [isOAuth2FlowInProgress, setOAuth2FlowInProgress] = useState(false);
  const [oAuthPopupController, setOAuthPopupController] =
    useState<AbortController | null>(null);
  const [oAuthError, setOAuthError] = useState<string | null>(null);

  if (!credentials) {
    return null;
  }

  if (credentials.isLoading) {
    return <div>Loading...</div>;
  }

  const {
    schema,
    provider,
    providerName,
    supportsApiKey,
    supportsOAuth2,
    savedApiKeys,
    savedOAuthCredentials,
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

  const ProviderIcon = providerIcons[provider];
  const modals = (
    <>
      {supportsApiKey && (
        <APIKeyCredentialsModal
          open={isAPICredentialsModalOpen}
          onClose={() => setAPICredentialsModalOpen(false)}
          onCredentialsCreate={(credsMeta) => {
            onSelectCredentials(credsMeta);
            setAPICredentialsModalOpen(false);
          }}
        />
      )}
      {supportsOAuth2 && (
        <OAuth2FlowWaitingModal
          open={isOAuth2FlowInProgress}
          onClose={() => oAuthPopupController?.abort("canceled")}
          providerName={providerName}
        />
      )}
    </>
  );

  // No saved credentials yet
  if (savedApiKeys.length === 0 && savedOAuthCredentials.length === 0) {
    return (
      <>
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
        </div>
        {modals}
        {oAuthError && (
          <div className="mt-2 text-red-500">Error: {oAuthError}</div>
        )}
      </>
    );
  }

  function handleValueChange(newValue: string) {
    if (newValue === "sign-in") {
      // Trigger OAuth2 sign in flow
      handleOAuthLogin();
    } else if (newValue === "add-api-key") {
      // Open API key dialog
      setAPICredentialsModalOpen(true);
    } else {
      const selectedCreds = savedApiKeys
        .concat(savedOAuthCredentials)
        .find((c) => c.id == newValue)!;

      onSelectCredentials({
        id: selectedCreds.id,
        type: selectedCreds.type,
        provider: schema.credentials_provider,
        // title: customTitle, // TODO: add input for title
      });
    }
  }

  // Saved credentials exist
  return (
    <>
      <Select value={selectedCredentials?.id} onValueChange={handleValueChange}>
        <SelectTrigger>
          <SelectValue placeholder={schema.placeholder} />
        </SelectTrigger>
        <SelectContent className="nodrag">
          {savedOAuthCredentials.map((credentials, index) => (
            <SelectItem key={index} value={credentials.id}>
              <ProviderIcon className="mr-2 inline h-4 w-4" />
              {credentials.username}
            </SelectItem>
          ))}
          {savedApiKeys.map((credentials, index) => (
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
        </SelectContent>
      </Select>
      {modals}
      {oAuthError && (
        <div className="mt-2 text-red-500">Error: {oAuthError}</div>
      )}
    </>
  );
};

export const APIKeyCredentialsModal: FC<{
  open: boolean;
  onClose: () => void;
  onCredentialsCreate: (creds: CredentialsMetaInput) => void;
}> = ({ open, onClose, onCredentialsCreate }) => {
  const credentials = useCredentials();

  const formSchema = z.object({
    apiKey: z.string().min(1, "API Key is required"),
    title: z.string().min(1, "Name is required"),
    expiresAt: z.string().optional(),
  });

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      apiKey: "",
      title: "",
      expiresAt: "",
    },
  });

  if (!credentials || credentials.isLoading || !credentials.supportsApiKey) {
    return null;
  }

  const { schema, provider, providerName, createAPIKeyCredentials } =
    credentials;

  async function onSubmit(values: z.infer<typeof formSchema>) {
    const expiresAt = values.expiresAt
      ? new Date(values.expiresAt).getTime() / 1000
      : undefined;
    const newCredentials = await createAPIKeyCredentials({
      api_key: values.apiKey,
      title: values.title,
      expires_at: expiresAt,
    });
    onCredentialsCreate({
      provider,
      id: newCredentials.id,
      type: "api_key",
      title: newCredentials.title,
    });
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add new API key for {providerName}</DialogTitle>
          {schema.description && (
            <DialogDescription>{schema.description}</DialogDescription>
          )}
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="apiKey"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>API Key</FormLabel>
                  {schema.credentials_scopes && (
                    <FormDescription>
                      Required scope(s) for this block:{" "}
                      {schema.credentials_scopes?.map((s, i, a) => (
                        <span key={i}>
                          <code>{s}</code>
                          {i < a.length - 1 && ", "}
                        </span>
                      ))}
                    </FormDescription>
                  )}
                  <FormControl>
                    <Input
                      type="password"
                      placeholder="Enter API key..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="title"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input
                      type="text"
                      placeholder="Enter a name for this API key..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="expiresAt"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Expiration Date (Optional)</FormLabel>
                  <FormControl>
                    <Input
                      type="datetime-local"
                      placeholder="Select expiration date..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button type="submit" className="w-full">
              Save & use this API key
            </Button>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  );
};

export const OAuth2FlowWaitingModal: FC<{
  open: boolean;
  onClose: () => void;
  providerName: string;
}> = ({ open, onClose, providerName }) => {
  return (
    <Dialog
      open={open}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>
            Waiting on {providerName} sign-in process...
          </DialogTitle>
          <DialogDescription>
            Complete the sign-in process in the pop-up window.
            <br />
            Closing this dialog will cancel the sign-in process.
          </DialogDescription>
        </DialogHeader>
      </DialogContent>
    </Dialog>
  );
};
