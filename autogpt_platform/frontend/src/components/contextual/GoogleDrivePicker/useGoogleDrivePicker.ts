import { getGetV1GetSpecificCredentialByIdQueryOptions } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { OAuth2Credentials } from "@/app/api/__generated__/models/oAuth2Credentials";
import { useToast } from "@/components/molecules/Toast/use-toast";
import useCredentials from "@/hooks/useCredentials";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  getCredentialsSchema,
  GooglePickerView,
  loadGoogleAPIPicker,
  loadGoogleIdentityServices,
  mapViewId,
  NormalizedPickedFile,
  normalizePickerResponse,
  scopesIncludeDrive,
} from "./helpers";

const defaultScopes = ["https://www.googleapis.com/auth/drive.file"];

type TokenClient = {
  requestAccessToken: (opts: { prompt: string }) => void;
};

export type Props = {
  scopes?: string[];
  developerKey?: string;
  clientId?: string;
  appId?: string; // Cloud project number
  multiselect?: boolean;
  views?: GooglePickerView[];
  navHidden?: boolean;
  listModeIfNoDriveScope?: boolean;
  disableThumbnails?: boolean;
  buttonText?: string;
  disabled?: boolean;
  /** When true, requires saved platform credentials (no consent flow fallback) */
  requirePlatformCredentials?: boolean;
  onPicked: (files: NormalizedPickedFile[], credentialId?: string) => void;
  onCanceled: () => void;
  onError: (err: unknown) => void;
};

export function useGoogleDrivePicker(options: Props) {
  const {
    scopes = ["https://www.googleapis.com/auth/drive.file"],
    developerKey = process.env.NEXT_PUBLIC_GOOGLE_API_KEY,
    clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID,
    appId = process.env.NEXT_PUBLIC_GOOGLE_APP_ID,
    multiselect = false,
    views = ["DOCS"],
    navHidden = false,
    listModeIfNoDriveScope = true,
    disableThumbnails = false,
    onPicked,
    onCanceled,
    onError,
  } = options || {};

  const requestedScopes = options?.scopes || defaultScopes;
  const [isLoading, setIsLoading] = useState(false);
  const [isAuthInProgress, setIsAuthInProgress] = useState(false);
  const [hasInsufficientScopes, setHasInsufficientScopes] = useState(false);
  const [selectedCredential, setSelectedCredential] = useState<
    CredentialsMetaInput | undefined
  >();
  const accessTokenRef = useRef<string | null>(null);
  const tokenClientRef = useRef<TokenClient | null>(null);
  const pickerReadyRef = useRef(false);
  const usedCredentialIdRef = useRef<string | undefined>(undefined);
  const credentials = useCredentials(getCredentialsSchema(requestedScopes));
  const queryClient = useQueryClient();
  const isReady = pickerReadyRef.current && !!tokenClientRef.current;
  const { toast } = useToast();

  const hasGoogleOAuth = useMemo(() => {
    if (!credentials || credentials.isLoading) return false;
    return credentials.savedCredentials?.length > 0;
  }, [credentials]);

  useEffect(() => {
    if (
      hasGoogleOAuth &&
      credentials &&
      !credentials.isLoading &&
      credentials.savedCredentials?.length > 0
    ) {
      setHasInsufficientScopes(false);
    }
  }, [hasGoogleOAuth, credentials]);

  useEffect(() => {
    if (
      credentials &&
      !credentials.isLoading &&
      credentials.savedCredentials?.length === 1 &&
      !selectedCredential
    ) {
      setSelectedCredential({
        id: credentials.savedCredentials[0].id,
        type: credentials.savedCredentials[0].type,
        provider: credentials.savedCredentials[0].provider,
        title: credentials.savedCredentials[0].title,
      });
    }
  }, [credentials, selectedCredential]);

  async function openPicker() {
    try {
      await ensureLoaded();

      if (
        hasGoogleOAuth &&
        credentials &&
        !credentials.isLoading &&
        credentials.savedCredentials?.length > 0
      ) {
        const credentialId =
          selectedCredential?.id || credentials.savedCredentials[0].id;
        usedCredentialIdRef.current = credentialId;

        try {
          const queryOptions = getGetV1GetSpecificCredentialByIdQueryOptions(
            "google",
            credentialId,
          );

          const response = await queryClient.fetchQuery(queryOptions);

          if (response.status === 200 && response.data) {
            const cred = response.data;
            if (cred.type === "oauth2") {
              const oauthCred = cred as OAuth2Credentials;
              if (oauthCred.access_token) {
                const credentialScopes = new Set(oauthCred.scopes || []);
                const requiredScopesSet = new Set(requestedScopes);
                const hasRequiredScopes = Array.from(requiredScopesSet).every(
                  (scope) => credentialScopes.has(scope),
                );

                if (!hasRequiredScopes) {
                  const error = new Error(
                    "The saved Google OAuth credentials do not have the required permissions. Please sign in again with the correct permissions.",
                  );
                  toast({
                    title: "Insufficient Permissions",
                    description: error.message,
                    variant: "destructive",
                  });
                  setHasInsufficientScopes(true);
                  if (onError) onError(error);
                  return;
                }

                accessTokenRef.current = oauthCred.access_token;
                buildAndShowPicker(oauthCred.access_token);
                return;
              }
            }
          }

          const error = new Error(
            "Failed to retrieve Google OAuth credentials. Please try signing in again.",
          );
          if (onError) onError(error);
          return;
        } catch (err) {
          const error =
            err instanceof Error
              ? err
              : new Error("Failed to fetch Google OAuth credentials");

          toast({
            title: "Authentication Error",
            description: error.message,
            variant: "destructive",
          });

          if (onError) onError(error);

          return;
        }
      }

      // If platform credentials are required but none exist, show error
      if (options?.requirePlatformCredentials) {
        const error = new Error(
          "Please connect your Google account in Settings before using this feature.",
        );
        toast({
          title: "Google Account Required",
          description: error.message,
          variant: "destructive",
        });
        if (onError) onError(error);
        return;
      }

      const token = accessTokenRef.current || (await requestAccessToken());
      buildAndShowPicker(token);
    } catch (e) {
      if (onError) onError(e);
    }
  }

  function ensureLoaded() {
    async function load() {
      try {
        setIsLoading(true);

        await Promise.all([
          loadGoogleAPIPicker(),
          loadGoogleIdentityServices(),
        ]);

        if (!clientId) throw new Error("Google OAuth client ID is not set");
        tokenClientRef.current =
          window.google!.accounts!.oauth2!.initTokenClient({
            client_id: clientId,
            scope: scopes.join(" "),
            callback: () => {},
          });

        pickerReadyRef.current = true;
      } catch (e) {
        console.error(e);
        toast({
          title: "Error loading Google Drive Picker",
          description: "Please try again later",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    }
    return load();
  }

  async function requestAccessToken() {
    function executor(
      resolve: (value: string) => void,
      reject: (reason?: unknown) => void,
    ) {
      const tokenClient = tokenClientRef.current;

      if (!tokenClient) {
        return reject(new Error("Token client not initialized"));
      }

      setIsAuthInProgress(true);
      // Update the callback on the already-initialized token client,
      // then request an access token using the token flow (no redirects).
      (tokenClient as any).callback = onTokenResponseFactory(resolve, reject);
      tokenClient.requestAccessToken({
        prompt: accessTokenRef.current ? "" : "consent",
      });
    }

    return await new Promise(executor);
  }

  function buildAndShowPicker(accessToken: string): void {
    if (!developerKey) {
      const error = new Error(
        "Missing Google Drive Picker Configuration: developer key is not set",
      );
      console.error("[useGoogleDrivePicker]", error.message);
      onError(error);
      return;
    }

    if (!appId) {
      const error = new Error(
        "Missing Google Drive Picker Configuration: app ID is not set",
      );
      console.error("[useGoogleDrivePicker]", error.message);
      onError(error);
      return;
    }

    const gp = window.google!.picker!;

    const builder = new gp.PickerBuilder()
      .setOAuthToken(accessToken)
      .setDeveloperKey(developerKey)
      .setAppId(appId)
      .setCallback(handlePickerData);

    if (navHidden) builder.enableFeature(gp.Feature.NAV_HIDDEN);
    if (multiselect) builder.enableFeature(gp.Feature.MULTISELECT_ENABLED);

    const allowThumbnails = disableThumbnails
      ? false
      : scopesIncludeDrive(scopes);

    views.forEach((v) => {
      const vid = mapViewId(v);
      const view = new gp.DocsView(vid);

      if (!allowThumbnails && listModeIfNoDriveScope) {
        view.setMode(gp.DocsViewMode.LIST);
      }

      builder.addView(view);
    });

    const picker = builder.build();

    // Mark picker as open - prevents parent dialogs from closing on outside clicks
    document.body.setAttribute("data-google-picker-open", "true");

    picker.setVisible(true);
  }

  function handlePickerData(data: any): void {
    // Google Picker fires callback on multiple events: LOADED, PICKED, CANCEL
    // Only remove the marker and process when picker is actually closed (PICKED or CANCEL)
    const gp = window.google?.picker;
    if (!gp || !data) return;

    const action = data[gp.Response.ACTION];

    // Ignore LOADED action - picker is still open
    // Note: gp.Action.LOADED exists at runtime but not in types
    if (action === "loaded") {
      return;
    }

    // Remove the marker when picker closes (PICKED or CANCEL)
    document.body.removeAttribute("data-google-picker-open");

    try {
      const files = normalizePickerResponse(data);
      if (files.length) {
        // Pass the credential ID that was used for this picker session
        onPicked(files, usedCredentialIdRef.current);
      } else {
        onCanceled();
      }
    } catch (e) {
      onError(e);
    }
  }

  function onTokenResponseFactory(
    resolve: (value: string) => void,
    reject: (reason?: unknown) => void,
  ) {
    return function onTokenResponse(response: any) {
      setIsAuthInProgress(false);
      if (response?.error) return reject(response);
      accessTokenRef.current = response.access_token;
      resolve(response.access_token);
    };
  }

  return {
    isReady,
    isLoading,
    isAuthInProgress,
    handleOpenPicker: openPicker,
    credentials,
    hasGoogleOAuth: hasInsufficientScopes ? false : hasGoogleOAuth,
    accessToken: accessTokenRef.current,
    selectedCredential,
    setSelectedCredential,
    usedCredentialId: usedCredentialIdRef.current,
  };
}
