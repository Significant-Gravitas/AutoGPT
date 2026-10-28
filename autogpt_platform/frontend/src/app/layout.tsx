import { fonts } from "@/components/styles/fonts";
import type { Metadata, Viewport } from "next";
import React from "react";

import "./globals.css";

import { Providers } from "@/app/providers";
import { ErrorBoundary } from "@/components/molecules/ErrorBoundary/ErrorBoundary";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { SetupAnalytics } from "@/services/analytics";
import { buildConsentDefaultsScript } from "@/services/analytics/consent-mode";
import { VercelSpeedInsights } from "@/services/analytics/VercelSpeedInsights";
import { ConsentWithdrawalReload } from "@/services/consent/ConsentWithdrawalReload";
import {
  COOKIEBOT_SCRIPT_ID,
  COOKIEBOT_SCRIPT_URL,
} from "@/services/consent/cookiebot";
import { environment } from "@/services/environment";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { headers } from "next/headers";
import Script from "next/script";
import { getSiteUrl } from "@/lib/metadata";

const isDev = environment.isDev();
const isLocal = environment.isLocal();

const faviconPath = isDev
  ? "/favicon-dev.ico"
  : isLocal
    ? "/favicon-local.ico"
    : "/favicon.ico";

const SITE_TITLE = "AutoGPT Platform";
const SITE_DESCRIPTION = "Your one stop shop to creating AI Agents";

export const metadata: Metadata = {
  metadataBase: new URL(getSiteUrl()),
  title: SITE_TITLE,
  description: SITE_DESCRIPTION,
  manifest: "/manifest.webmanifest",
  icons: {
    icon: faviconPath,
    apple: "/apple-touch-icon.png",
  },
  openGraph: {
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    siteName: "AutoGPT",
    type: "website",
  },
  twitter: {
    card: "summary",
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // Android shrinks the page for the on-screen keyboard instead of sliding
  // it over the bottom, so the chat composer stays in view above it.
  interactiveWidget: "resizes-content",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const headersList = await headers();
  const host = headersList.get("host") || "";
  const cookiebotCBID = environment.getCookiebotCBID();
  const cookiebotGeoRegions = environment.getCookiebotGeoRegions();

  return (
    <html
      lang="en"
      className={`${fonts.poppins.variable} ${fonts.sans.variable} ${fonts.mono.variable}`}
      suppressHydrationWarning
    >
      {/* Mobile Safari's 100vh is the height with its toolbar hidden, so a
          min-h-screen page always scrolls a little; dvh is what is visible.
          min-h-screen stays as the fallback for browsers without dvh. */}
      <body className="min-h-screen supports-[min-height:100dvh]:min-h-dvh">
        {process.env.NEXT_PUBLIC_RENDER_COUNTER === "1" ? (
          <Script
            src="/devtools/render-counter.js"
            strategy="beforeInteractive"
          />
        ) : null}
        {/* Without a Cookiebot domain group there is no banner and every
            optional category stays denied. With one, the Consent Mode
            defaults are queued before anything else can reach the Google
            tag. Cookiebot itself loads after hydration so a slow CDN cannot
            hold the app back: blocking stays manual (the tools we gate are
            bundled npm SDKs its auto-blocking cannot see), every one of them
            asks services/consent, and until uc.js arrives that reads the same
            stored answer uc.js reads when it starts. */}
        {cookiebotCBID ? (
          <>
            <Script
              id="google-consent-defaults"
              strategy="beforeInteractive"
              data-cookieconsent="ignore"
              dangerouslySetInnerHTML={{ __html: buildConsentDefaultsScript() }}
            />
            <Script
              id={COOKIEBOT_SCRIPT_ID}
              src={COOKIEBOT_SCRIPT_URL}
              data-cbid={cookiebotCBID}
              data-georegions={cookiebotGeoRegions || undefined}
              data-blockingmode="manual"
              strategy="afterInteractive"
            />
          </>
        ) : null}
        <ErrorBoundary context="application">
          <Providers
            attribute="class"
            defaultTheme="light"
            // Feel free to remove this line if you want to use the system theme by default
            // enableSystem
            disableTransitionOnChange
          >
            <SetupAnalytics
              host={host}
              ga={{ gaId: environment.getGAMeasurementID() }}
            />
            <div className="flex min-h-screen flex-col items-stretch justify-items-stretch supports-[min-height:100dvh]:min-h-dvh">
              {children}
              <VercelSpeedInsights />

              {/* React Query DevTools is only available in development */}
              {process.env.NEXT_PUBLIC_REACT_QUERY_DEVTOOL && (
                <ReactQueryDevtools
                  initialIsOpen={false}
                  buttonPosition={"bottom-left"}
                />
              )}
            </div>
            <Toaster />
            <ConsentWithdrawalReload />
          </Providers>
        </ErrorBoundary>
      </body>
    </html>
  );
}
