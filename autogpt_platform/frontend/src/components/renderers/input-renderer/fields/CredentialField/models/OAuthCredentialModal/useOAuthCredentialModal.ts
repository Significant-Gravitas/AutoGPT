import {
  getGetV1ListCredentialsQueryKey,
  useGetV1InitiateOauthFlow,
  usePostV1ExchangeOauthCodeForTokens,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { LoginResponse } from "@/app/api/__generated__/models/loginResponse";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

type useOAuthCredentialModalProps = {
  provider: string;
  scopes?: string[];
};

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

export const useOAuthCredentialModal = ({
  provider,
  scopes,
}: useOAuthCredentialModalProps) => {
  const { toast } = useToast();

  const [open, setOpen] = useState(false);
  const [oAuthPopupController, setOAuthPopupController] =
    useState<AbortController | null>(null);
  const [oAuthError, setOAuthError] = useState<string | null>(null);
  const [isOAuth2FlowInProgress, setOAuth2FlowInProgress] = useState(false);

  const queryClient = useQueryClient();

  const {
    refetch: initiateOauthFlow,
    isRefetching: isInitiatingOauthFlow,
    isRefetchError: initiatingOauthFlowError,
  } = useGetV1InitiateOauthFlow(
    provider,
    {
      scopes: scopes?.join(","),
    },
    {
      query: {
        enabled: false,
        select: (res) => {
          return res.data as LoginResponse;
        },
      },
    },
  );

  const {
    mutateAsync: oAuthCallback,
    isPending: isOAuthCallbackPending,
    error: oAuthCallbackError,
  } = usePostV1ExchangeOauthCodeForTokens({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetV1ListCredentialsQueryKey(),
        });
        setOpen(false);
        toast({
          title: "Success",
          description: "Credential added successfully",
          variant: "default",
        });
      },
    },
  });

  const handleOAuthLogin = async () => {
    const { data } = await initiateOauthFlow();
    if (!data || !data.login_url || !data.state_token) {
      toast({
        title: "Failed to initiate OAuth flow",
        variant: "destructive",
      });
      setOAuthError(
        data && typeof data === "object" && "detail" in data
          ? (data.detail as string)
          : "Failed to initiate OAuth flow",
      );
      return;
    }

    setOpen(true);
    setOAuth2FlowInProgress(true);

    const { login_url, state_token } = data;

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
        await oAuthCallback({
          provider,
          data: {
            code: e.data.code,
            state_token: e.data.state,
          },
        });

        console.debug("OAuth callback processed successfully");
      } catch (error) {
        console.error("Error in OAuth callback:", error);
        setOAuthError(
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
  };

  const onClose = () => {
    oAuthPopupController?.abort("canceled");
    setOpen(false);
  };

  return {
    handleOAuthLogin,
    loading:
      isOAuth2FlowInProgress || isOAuthCallbackPending || isInitiatingOauthFlow,
    error: oAuthError || initiatingOauthFlowError || oAuthCallbackError,
    onClose,
    open,
    setOpen,
  };
};
