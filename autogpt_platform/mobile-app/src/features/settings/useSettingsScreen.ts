import { useQuery } from "@tanstack/react-query";
import { useRef } from "react";
import { Linking } from "react-native";
import type { WebView } from "react-native-webview";
import type {
  ShouldStartLoadRequest,
  WebViewNavigation,
} from "react-native-webview/lib/WebViewTypes";

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
  const webViewRef = useRef<WebView<object>>(null);
  // One page change can report several states; go back only once for it.
  const isGoingBack = useRef(false);

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

  // Next.js links change the page with pushState, which never reaches
  // handleNavigation, so a page outside Settings is undone here instead.
  function handleStateChange({ url, canGoBack }: WebViewNavigation) {
    if (url === "about:blank" || isAllowedSettingsURL(url)) {
      isGoingBack.current = false;
      return;
    }
    if (!canGoBack || isGoingBack.current) return;
    isGoingBack.current = true;
    webViewRef.current?.goBack();
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
    webViewRef,
    handleNavigation,
    handleStateChange,
  };
}
