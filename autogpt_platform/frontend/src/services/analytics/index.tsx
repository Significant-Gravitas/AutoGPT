/**
 * Modified copy of ga.tsx from @next/third-parties/google, with modified gtag.js source URL.
 * Original source file: https://github.com/vercel/next.js/blob/b304b45e3a6e3e79338568d76e28805e77c03ec9/packages/third-parties/src/google/ga.tsx
 */

"use client";

import type { GAParams } from "@/types/google";
import { consent } from "@/services/consent/cookies";
import Script from "next/script";
import { useEffect, useState } from "react";
import { environment } from "../environment";

declare global {
  interface Window {
    datafast: (name: string, metadata: Record<string, unknown>) => void;
    [key: string]: unknown[] | ((...args: unknown[]) => void) | unknown;
  }
}

let currDataLayerName: string | undefined = undefined;

type SetupProps = {
  ga: GAParams;
  host: string;
};

export function SetupAnalytics(props: SetupProps) {
  const { ga, host } = props;
  const { gaId, debugMode, dataLayerName = "dataLayer", nonce } = ga;
  const isProductionDomain = host.includes("platform.agpt.co");

  // Check for user consent
  const [hasAnalyticsConsent, setHasAnalyticsConsent] = useState(false);

  useEffect(() => {
    // Check consent on mount
    setHasAnalyticsConsent(consent.hasConsentFor("analytics"));
  }, []);

  // Datafa.st journey analytics only on production AND with consent
  const dataFastEnabled = isProductionDomain && hasAnalyticsConsent;
  // We collect analytics too for open source developers running the platform locally
  // BUT only with consent
  const googleAnalyticsEnabled =
    (environment.isLocal() || isProductionDomain) && hasAnalyticsConsent;

  if (currDataLayerName === undefined) {
    currDataLayerName = dataLayerName;
  }

  useEffect(() => {
    if (!googleAnalyticsEnabled) return;

    // Google Analytics: feature usage signal (same as original implementation)
    performance.mark("mark_feature_usage", {
      detail: {
        feature: "custom-ga",
      },
    });
  }, [googleAnalyticsEnabled]);

  return (
    <>
      {/* Google Analytics */}
      {googleAnalyticsEnabled ? (
        <>
          <Script
            id="_custom-ga-init"
            strategy="afterInteractive"
            dangerouslySetInnerHTML={{
              __html: `
            window['${dataLayerName}'] = window['${dataLayerName}'] || [];
            function gtag(){window['${dataLayerName}'].push(arguments);}
            gtag('js', new Date());
            gtag('config', '${gaId}' ${debugMode ? ",{ 'debug_mode': true }" : ""});
          `,
            }}
            nonce={nonce}
          />
          {/* Google Tag Manager */}
          <Script
            id="_custom-ga"
            strategy="afterInteractive"
            src="/gtag.js"
            nonce={nonce}
          />
        </>
      ) : null}
      {/* Datafa.st */}
      {dataFastEnabled ? (
        <Script
          strategy="afterInteractive"
          data-website-id="dfid_g5wtBIiHUwSkWKcGz80lu"
          data-domain="agpt.co"
          src="https://datafa.st/js/script.js"
        />
      ) : null}
    </>
  );
}

export const analytics = {
  sendGAEvent,
  sendDatafastEvent,
};

function sendGAEvent(...args: unknown[]) {
  if (typeof window === "undefined") return;
  if (currDataLayerName === undefined) return;

  const dataLayer = window[currDataLayerName];
  if (!dataLayer) return;

  if (Array.isArray(dataLayer)) {
    dataLayer.push(...args);
  } else {
    console.warn(`Custom GA: dataLayer ${currDataLayerName} does not exist`);
  }
}

function sendDatafastEvent(name: string, metadata: Record<string, unknown>) {
  if (typeof window === "undefined" || !window.datafast) return;
  window.datafast(name, metadata);
}
