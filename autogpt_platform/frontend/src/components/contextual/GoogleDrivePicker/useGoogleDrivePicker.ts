import { useMemo, useRef, useState } from "react";
import {
  GooglePickerView,
  loadGoogleAPIPicker,
  loadGoogleIdentityServices,
  mapViewId,
  normalizePickerResponse,
  NormalizedPickedFile,
  scopesIncludeDrive,
} from "./helpers";

type TokenClient = {
  requestAccessToken: (opts: { prompt: string }) => void;
};

export type UseGoogleDrivePickerOptions = {
  scopes?: string[];
  developerKey?: string;
  clientId?: string;
  appId?: string; // Cloud project number
  multiselect?: boolean;
  views?: GooglePickerView[];
  navHidden?: boolean;
  listModeIfNoDriveScope?: boolean;
  disableThumbnails?: boolean;
  onPicked?: (files: NormalizedPickedFile[]) => void;
  onCanceled?: () => void;
  onError?: (err: unknown) => void;
};

export function useGoogleDrivePicker(options?: UseGoogleDrivePickerOptions) {
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

  const [isLoading, setIsLoading] = useState(false);
  const [isAuthInProgress, setIsAuthInProgress] = useState(false);
  const accessTokenRef = useRef<string | null>(null);
  const tokenClientRef = useRef<TokenClient | null>(null);
  const pickerReadyRef = useRef(false);

  const isReady = useMemo(
    () => pickerReadyRef.current && !!tokenClientRef.current,
    [],
  );

  function ensureLoaded(): Promise<void> {
    async function load(): Promise<void> {
      try {
        setIsLoading(true);
        await Promise.all([
          loadGoogleAPIPicker(),
          loadGoogleIdentityServices(),
        ]);
        const googleObj = (window as any).google;
        tokenClientRef.current = googleObj.accounts.oauth2.initTokenClient({
          client_id: clientId,
          scope: scopes.join(" "),
          callback: noop,
        });
        pickerReadyRef.current = true;
      } finally {
        setIsLoading(false);
      }
    }
    return load();
  }

  function handlePickerData(data: any): void {
    try {
      const files = normalizePickerResponse(data);
      if (files.length) {
        if (onPicked) onPicked(files);
      } else if (onCanceled) {
        onCanceled();
      }
    } catch (e) {
      if (onError) onError(e);
    }
  }

  function buildAndShowPicker(accessToken: string): void {
    const gp = (window as any).google.picker;
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
      if (!allowThumbnails || listModeIfNoDriveScope)
        view.setMode(gp.DocsViewMode.LIST);
      builder.addView(view);
    });

    const picker = builder.build();
    picker.setVisible(true);
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

  async function requestAccessToken(): Promise<string> {
    function executor(
      resolve: (value: string) => void,
      reject: (reason?: unknown) => void,
    ) {
      const tokenClient = tokenClientRef.current;
      if (!tokenClient)
        return reject(new Error("Token client not initialized"));

      setIsAuthInProgress(true);
      const googleObj = (window as any).google;
      googleObj.accounts.oauth2
        .initTokenClient({
          client_id: clientId,
          scope: scopes.join(" "),
          callback: onTokenResponseFactory(resolve, reject),
        })
        .requestAccessToken({
          prompt: accessTokenRef.current ? "" : "consent",
        });
    }

    return await new Promise<string>(executor);
  }

  async function openPicker() {
    try {
      await ensureLoaded();
      const token = accessTokenRef.current || (await requestAccessToken());
      buildAndShowPicker(token);
    } catch (e) {
      if (onError) onError(e);
    }
  }

  return {
    isReady,
    isLoading,
    isAuthInProgress,
    openPicker,
  };
}

function noop() {}
