import { useRef, useState } from "react";
import { Linking } from "react-native";
import type { WebView } from "react-native-webview";
import type {
  ShouldStartLoadRequest,
  WebViewNavigation,
} from "react-native-webview/lib/WebViewTypes";

import { useSession } from "@/lib/auth/nativeSession";
import { sessionCookieHeader } from "@/lib/auth/sessionHelpers";
import { FRONTEND_URL } from "@/lib/env";

import {
  cookieScript,
  isAllowedSettingsURL,
  isFrontendOrigin,
  isWebURL,
} from "./helpers";

export function useSettingsScreen() {
  const session = useSession();
  const cookie = session
    ? sessionCookieHeader(session.token, FRONTEND_URL)
    : "";
  const webViewRef = useRef<WebView<object>>(null);
  // One page change can report several states; go back only once for it.
  const isGoingBack = useRef(false);
  const hasShownSettings = useRef(false);
  const [isBlocked, setIsBlocked] = useState(false);
  const [webViewKey, setWebViewKey] = useState(0);

  function handleNavigation(request: ShouldStartLoadRequest) {
    const { url } = request;
    if (url === "about:blank") return true;
    // iOS also asks about iframes (payment forms and the like); they load in
    // place and must never send the person to the browser.
    if (request.isTopFrame === false) return isWebURL(url);
    if (isAllowedSettingsURL(url)) return true;
    if (isWebURL(url) && !isFrontendOrigin(url)) {
      Linking.openURL(url).catch(() => undefined);
    } else if (isFrontendOrigin(url) && !hasShownSettings.current) {
      setIsBlocked(true);
    }
    return false;
  }

  // Next.js links change the page with pushState, which never reaches
  // handleNavigation, so a page outside Settings is undone here instead.
  function handleStateChange({ url, canGoBack, loading }: WebViewNavigation) {
    if (url === "about:blank") {
      isGoingBack.current = false;
      return;
    }
    if (isAllowedSettingsURL(url)) {
      isGoingBack.current = false;
      // Load start reports the URL too, before a redirect away from it.
      if (!loading) hasShownSettings.current = true;
      return;
    }
    if (!canGoBack) {
      setIsBlocked(true);
      return;
    }
    if (isGoingBack.current) return;
    isGoingBack.current = true;
    webViewRef.current?.goBack();
  }

  function retry() {
    setIsBlocked(false);
    hasShownSettings.current = false;
    setWebViewKey((key) => key + 1);
  }

  return {
    settingsURL: `${FRONTEND_URL}/settings`,
    cookie,
    script: cookieScript(cookie, FRONTEND_URL),
    isBlocked,
    webViewKey,
    retry,
    webViewRef,
    handleNavigation,
    handleStateChange,
  };
}
