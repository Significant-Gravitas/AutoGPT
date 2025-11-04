import { fonts } from "@/components/styles/fonts";
import type { Metadata } from "next";
import React from "react";

import "./globals.css";

import { Providers } from "@/app/providers";
import { CookieConsentBanner } from "@/components/molecules/CookieConsentBanner/CookieConsentBanner";
import TallyPopupSimple from "@/components/molecules/TallyPoup/TallyPopup";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { headers } from "next/headers";
import { SetupAnalytics } from "@/services/analytics";
import { VercelAnalyticsWrapper } from "@/services/analytics/VercelAnalyticsWrapper";

export const metadata: Metadata = {
  title: "AutoGPT Platform",
  description: "Your one stop shop to creating AI Agents",
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
      <head>
        <SetupAnalytics
          host={host}
          ga={{
            gaId: process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID || "G-FH2XK2W4GN",
          }}
        />
      </head>
      <body>
        <Providers
          attribute="class"
          defaultTheme="light"
          // Feel free to remove this line if you want to use the system theme by default
          // enableSystem
          disableTransitionOnChange
        >
          <div className="flex min-h-screen flex-col items-stretch justify-items-stretch">
            {children}
            <TallyPopupSimple />
            <VercelAnalyticsWrapper />

            {/* React Query DevTools is only available in development */}
            {process.env.NEXT_PUBLIC_REACT_QUERY_DEVTOOL && (
              <ReactQueryDevtools
                initialIsOpen={false}
                buttonPosition={"bottom-left"}
              />
            )}
          </div>
          <Toaster />
          <CookieConsentBanner />
        </Providers>
      </body>
    </html>
  );
}
