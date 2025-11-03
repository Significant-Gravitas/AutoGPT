import { useToast } from "@/components/molecules/Toast/use-toast";
import useCredentials from "@/hooks/useCredentials";
import { useMemo, useRef, useState } from "react";
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
  onPicked: (files: NormalizedPickedFile[]) => void;
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
  const accessTokenRef = useRef<string | null>(null);
  const tokenClientRef = useRef<TokenClient | null>(null);
  const pickerReadyRef = useRef(false);
  const credentials = useCredentials(getCredentialsSchema(requestedScopes));
  const isReady = pickerReadyRef.current && !!tokenClientRef.current;
  const { toast } = useToast();

  const hasGoogleOAuth = useMemo(() => {
    if (!credentials || credentials.isLoading) return false;
    return credentials.savedCredentials?.length > 0;
  }, [credentials]);

  async function openPicker() {
    try {
      await ensureLoaded();
      console.log(accessTokenRef.current);
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
    picker.setVisible(true);
  }

  function handlePickerData(data: any): void {
    try {
      const files = normalizePickerResponse(data);
      if (files.length) {
        onPicked(files);
      } else {
        onCanceled();
      }
    } catch (e) {
      if (onError) onError(e);
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
    hasGoogleOAuth,
  };
}
