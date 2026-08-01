import { fonts } from "@/components/styles/fonts";
import type { Metadata } from "next";
import React from "react";

import "./globals.css";

import { Providers } from "@/app/providers";
import { CookieConsentBanner } from "@/components/molecules/CookieConsentBanner/CookieConsentBanner";
import { ErrorBoundary } from "@/components/molecules/ErrorBoundary/ErrorBoundary";
import { TallyPopupProvider } from "@/components/molecules/TallyPoup/TallyPopup";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { SetupAnalytics } from "@/services/analytics";
import { VercelAnalyticsWrapper } from "@/services/analytics/VercelAnalyticsWrapper";
import { environment } from "@/services/environment";
import AgentationDevtool from "@/components/AgentationDevtool";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { headers } from "next/headers";

const isDev = environment.isDev();
const isLocal = environment.isLocal();
const telemetryEnabled =
  !isLocal || process.env.AUTOGPT_TELEMETRY_ENABLED === "true";
const feedbackEnabled =
  !isLocal || process.env.AUTOGPT_FEEDBACK_ENABLED === "true";
const developerUiEnabled =
  isDev || process.env.AUTOGPT_DEVELOPER_UI_ENABLED === "true";

const faviconPath = isDev
  ? "/favicon-dev.ico"
  : isLocal
    ? "/favicon-local.ico"
    : "/favicon.ico";

export const metadata: Metadata = {
  title: "AutoGPT Platform",
  description: "Your one stop shop to creating AI Agents",
  manifest: "/manifest.webmanifest",
  icons: {
    icon: faviconPath,
    apple: "/apple-touch-icon.png",
  },
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const headersList = await headers();
  const host = headersList.get("host") || "";

  return (
    <html
      lang="en"
      className={`${fonts.poppins.variable} ${fonts.sans.variable} ${fonts.mono.variable}`}
      suppressHydrationWarning
    >
      <body className="min-h-screen">
        <ErrorBoundary context="application">
          <Providers
            attribute="class"
            defaultTheme="light"
            // Feel free to remove this line if you want to use the system theme by default
            // enableSystem
            disableTransitionOnChange
          >
            <TallyPopupProvider enabled={feedbackEnabled}>
              <SetupAnalytics
                enabled={telemetryEnabled}
                host={host}
                ga={{
                  gaId:
                    process.env.AUTOGPT_GA_MEASUREMENT_ID ||
                    process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID ||
                    "G-FH2XK2W4GN",
                }}
              />
              <div className="flex min-h-screen flex-col items-stretch justify-items-stretch">
                {children}
                <VercelAnalyticsWrapper enabled={telemetryEnabled} />

                {/* React Query DevTools is only available in development */}
                {developerUiEnabled &&
                  process.env.NEXT_PUBLIC_REACT_QUERY_DEVTOOL === "true" && (
                    <ReactQueryDevtools
                      initialIsOpen={false}
                      buttonPosition={"bottom-left"}
                    />
                  )}
              </div>
              <Toaster />
              <CookieConsentBanner />
              {developerUiEnabled && <AgentationDevtool />}
            </TallyPopupProvider>
          </Providers>
        </ErrorBoundary>
      </body>
    </html>
  );
}
