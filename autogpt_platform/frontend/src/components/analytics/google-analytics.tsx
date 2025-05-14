/**
 * Modified copy of ga.tsx from @next/third-parties/google, with modified gtag.js source URL.
 * Original source file: https://github.com/vercel/next.js/blob/b304b45e3a6e3e79338568d76e28805e77c03ec9/packages/third-parties/src/google/ga.tsx
 */

"use client";

import { useEffect } from "react";
import Script from "next/script";
import type { GAParams } from "@/types/google";

let currDataLayerName: string | undefined = undefined;

export function GoogleAnalytics(props: GAParams) {
  const { gaId, debugMode, dataLayerName = "dataLayer", nonce } = props;

  if (currDataLayerName === undefined) {
    currDataLayerName = dataLayerName;
  }

  useEffect(() => {
    // Feature usage signal (same as original implementation)
    performance.mark("mark_feature_usage", {
      detail: {
        feature: "custom-ga",
      },
    });
  }, []);

  return (
    <>
      <Script
        id="_custom-ga-init"
        strategy="beforeInteractive"
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
      <Script
        id="_custom-ga"
        strategy="beforeInteractive"
        src="/gtag.js"
        nonce={nonce}
      />
    </>
  );
}

export function sendGAEvent(..._args: any[]) {
  if (currDataLayerName === undefined) {
    console.warn(`Custom GA: GA has not been initialized`);
    return;
  }
  //@ts-ignore
  if (window[currDataLayerName]) {
    //@ts-ignore
    window[currDataLayerName].push(arguments);
  } else {
    console.warn(`Custom GA: dataLayer ${currDataLayerName} does not exist`);
  }
}
