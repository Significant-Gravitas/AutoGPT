import { useQuery } from "@tanstack/react-query";
import { Linking } from "react-native";
import type { ShouldStartLoadRequest } from "react-native-webview/lib/WebViewTypes";

import { authClient } from "@/lib/auth/client";
import { FRONTEND_URL } from "@/lib/env";

import {
  cookieScript,
  isAllowedSettingsURL,
  isFrontendOrigin,
  isWebURL,
} from "./helpers";

const isSecure = FRONTEND_URL.startsWith("https");

export function useSettingsScreen() {
  const cookie = useQuery({
    queryKey: ["auth", "cookie"],
    queryFn: () => authClient.getCookie(),
    staleTime: 0,
    gcTime: 0,
  });

  function handleNavigation(request: ShouldStartLoadRequest) {
    const { url } = request;
    if (url === "about:blank") return true;
    // iOS also asks about iframes (payment forms and the like); they load in
    // place and must never send the person to the browser.
    if (request.isTopFrame === false) return isWebURL(url);
    if (isAllowedSettingsURL(url)) return true;
    if (isWebURL(url) && !isFrontendOrigin(url)) {
      Linking.openURL(url).catch(() => undefined);
    }
    return false;
  }

  function retry() {
    void cookie.refetch();
  }

  return {
    settingsURL: `${FRONTEND_URL}/settings`,
    cookie: cookie.data,
    script:
      cookie.data === undefined ? "" : cookieScript(cookie.data, isSecure),
    isError: cookie.isError,
    retry,
    handleNavigation,
  };
}
