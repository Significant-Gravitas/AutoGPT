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
import { connection } from "next/server";
import { resolveRuntimeControls } from "./runtimeControls";

const isDev = environment.isDev();
const isLocal = environment.isLocal();

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
  await connection();
  const runtimeControls = resolveRuntimeControls({
    publicOrigin:
      process.env.BETTER_AUTH_URL || process.env.NEXT_PUBLIC_FRONTEND_BASE_URL,
    isDev: isDev || environment.isDevelopmentBuild(),
    env: {
      AUTOGPT_TELEMETRY_ENABLED: process.env.AUTOGPT_TELEMETRY_ENABLED,
      AUTOGPT_FEEDBACK_ENABLED: process.env.AUTOGPT_FEEDBACK_ENABLED,
      AUTOGPT_DEVELOPER_UI_ENABLED: process.env.AUTOGPT_DEVELOPER_UI_ENABLED,
      AUTOGPT_GA_MEASUREMENT_ID: process.env.AUTOGPT_GA_MEASUREMENT_ID,
      NEXT_PUBLIC_GA_MEASUREMENT_ID: process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID,
    },
  });

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
            <TallyPopupProvider enabled={runtimeControls.feedbackEnabled}>
              <SetupAnalytics
                enabled={runtimeControls.telemetryEnabled}
                dataFastEnabled={runtimeControls.hostedPlatform}
                ga={
                  runtimeControls.gaMeasurementId
                    ? { gaId: runtimeControls.gaMeasurementId }
                    : undefined
                }
              />
              <div className="flex min-h-screen flex-col items-stretch justify-items-stretch">
                {children}
                <VercelAnalyticsWrapper
                  enabled={runtimeControls.hostedPlatform}
                />

                {process.env.NEXT_PUBLIC_REACT_QUERY_DEVTOOL === "true" && (
                  <ReactQueryDevtools
                    initialIsOpen={false}
                    buttonPosition={"bottom-left"}
                  />
                )}
              </div>
              <Toaster />
              <CookieConsentBanner />
              {runtimeControls.developerUiEnabled && <AgentationDevtool />}
            </TallyPopupProvider>
          </Providers>
        </ErrorBoundary>
      </body>
    </html>
  );
}
