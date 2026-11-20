/**
 * Modified copy of ga.tsx from @next/third-parties/google, with modified gtag.js source URL.
 * Original source file: https://github.com/vercel/next.js/blob/b304b45e3a6e3e79338568d76e28805e77c03ec9/packages/third-parties/src/google/ga.tsx
 */

"use client";

import type { GAParams } from "@/types/google";
import { hasConsentFor } from "@/services/consent/consent";
import Script from "next/script";
import { environment } from "../environment";
import { GoogleConsentModeSync } from "./GoogleConsentModeSync";
import { DATA_LAYER_NAME, gtag } from "./gtag";
import { DATAFAST_SCRIPT_SRC } from "./loading-policy";
import { useSetupAnalytics } from "./useSetupAnalytics";

type DatafastEvent = [name: string, metadata: Record<string, unknown>];

declare global {
  interface Window {
    datafast?: (...event: DatafastEvent) => void;
  }
}

type SetupProps = {
  ga: GAParams;
  host: string;
};

export function SetupAnalytics(props: SetupProps) {
  const { ga, host } = props;
  const { gaId, debugMode, nonce } = ga;
  const adsID = environment.getGoogleAdsID();
  const { googleTagEnabled, dataFastEnabled } = useSetupAnalytics(host);

  return (
    <>
      {/* Google tag: GA4 + Google Ads. Nothing to load without an ID: a
          self-hosted install must not report to anyone else's property */}
      {googleTagEnabled && (gaId || adsID) ? (
        <>
          <GoogleConsentModeSync />
          <Script
            id="_custom-ga-init"
            strategy="afterInteractive"
            dangerouslySetInnerHTML={{
              __html: buildGoogleTagInitScript({
                GAID: gaId,
                adsID,
                debugMode,
              }),
            }}
            nonce={nonce}
          />
          <Script
            id="_custom-ga"
            strategy="afterInteractive"
            src="/gtag.js"
            nonce={nonce}
          />
        </>
      ) : null}
      {/* DataFast is kept for attribution only: the script records the visit
          and sets the visitor and session cookies that the server-side signup
          goal, the X-Datafast-* headers and the Stripe checkout metadata read.
          In-app behaviour goes to PostHog. onLoad is load-bearing: it delivers
          the /tour events queued before the script finished loading */}
      {dataFastEnabled ? (
        <Script
          strategy="afterInteractive"
          data-website-id="dfid_g5wtBIiHUwSkWKcGz80lu"
          data-domain="agpt.co"
          src={DATAFAST_SCRIPT_SRC}
          onLoad={flushDatafastQueue}
        />
      ) : null}
    </>
  );
}

interface InitScriptArgs {
  GAID: string;
  adsID: string;
  debugMode?: boolean;
}

// The root layout queued the Consent Mode defaults before anything else, and
// GoogleConsentModeSync (with Cookiebot's own integration) sends the answer.
function buildGoogleTagInitScript({
  GAID,
  adsID,
  debugMode,
}: InitScriptArgs): string {
  // The IDs come from env vars and go into a nonce-bearing inline script, so
  // they are escaped rather than interpolated raw: a stray quote in a misfilled
  // env var would otherwise become CSP-blessed script.
  return [
    `window['${DATA_LAYER_NAME}'] = window['${DATA_LAYER_NAME}'] || [];`,
    `function gtag(){window['${DATA_LAYER_NAME}'].push(arguments);}`,
    `gtag('js', new Date());`,
    GAID
      ? `gtag('config', ${JSON.stringify(GAID)}${debugMode ? ", { 'debug_mode': true }" : ""});`
      : "",
    adsID
      ? `gtag('config', ${JSON.stringify(adsID)}, { 'allow_enhanced_conversions': true });`
      : "",
  ].join("\n");
}

export const analytics = {
  sendGAEvent,
  sendDatafastEvent,
};

function sendGAEvent(...args: unknown[]) {
  gtag(...args);
}

// Module scope means the queue survives client-side navigation: queued events
// are delivered wherever the script loads next, so metadata must never carry
// PII. On overflow the earliest events win — funnel starts fire first. The cap
// bounds memory when the script never loads (ad blockers, non-production
// domains where dataFastEnabled is false).
const MAX_QUEUED_DATAFAST_EVENTS = 100;
const datafastQueue: DatafastEvent[] = [];
let datafastQueueOverflowWarned = false;

// Returns whether the event was sent or queued, so a once-per-session guard
// can stay unset when the event was dropped.
function sendDatafastEvent(
  name: string,
  metadata: Record<string, unknown>,
): boolean {
  if (typeof window === "undefined") return false;
  // Checked on every event, not only before the script loads: the script
  // stays on window after consent is taken back. Pre-consent events must not
  // queue either — they would be replayed once consent is granted.
  if (!hasConsentFor("analytics")) return false;
  if (window.datafast) {
    // Self-heal if the Script's onLoad never fired (e.g. consent toggle
    // remounted it after the script had already loaded): replay the backlog
    // first so queued events keep their order ahead of this one.
    flushDatafastQueue();
    window.datafast(name, metadata);
    return true;
  }
  // The script loads afterInteractive, so mount-time events (tour_start,
  // tour_scenario_start) fire before window.datafast exists. Queue them and
  // flush from the Script's onLoad instead of dropping them.
  if (datafastQueue.length >= MAX_QUEUED_DATAFAST_EVENTS) {
    if (!datafastQueueOverflowWarned) {
      datafastQueueOverflowWarned = true;
      console.warn(
        `DataFast queue full (${MAX_QUEUED_DATAFAST_EVENTS} events); dropping new events until the script loads`,
      );
    }
    return false;
  }
  datafastQueue.push([name, metadata]);
  return true;
}

export function flushDatafastQueue() {
  if (typeof window === "undefined") return;
  if (!hasConsentFor("analytics")) {
    datafastQueue.length = 0;
    datafastQueueOverflowWarned = false;
    return;
  }
  if (!window.datafast) return;
  const datafast = window.datafast;
  datafastQueue
    .splice(0, datafastQueue.length)
    .forEach(([name, metadata]) => datafast(name, metadata));
  // A successful drain ends the blocked episode; warn again if a later one
  // overflows too.
  datafastQueueOverflowWarned = false;
}
