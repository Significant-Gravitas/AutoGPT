import { fonts } from "@/components/styles/fonts";
import type { Metadata } from "next";
import React from "react";

import "./globals.css";

import { Providers } from "@/app/providers";
import TallyPopupSimple from "@/components/molecules/TallyPoup/TallyPopup";
import { GoogleAnalytics } from "@/services/analytics/google-analytics";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Analytics } from "@vercel/analytics/next";

export const metadata: Metadata = {
  title: "AutoGPT Platform",
  description: "Your one stop shop to creating AI Agents",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${fonts.poppins.variable} ${fonts.sans.variable} ${fonts.mono.variable}`}
      suppressHydrationWarning
    >
      <head>
        <GoogleAnalytics
          gaId={process.env.NEXT_PUBLIC_GA_MEASUREMENT_ID || "G-FH2XK2W4GN"} // This is the measurement Id for the Google Analytics dev project
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
            <SpeedInsights />
            <Analytics />

            {/* React Query DevTools is only available in development */}
            {process.env.NEXT_PUBLIC_REACT_QUERY_DEVTOOL && (
              <ReactQueryDevtools
                initialIsOpen={false}
                buttonPosition={"bottom-left"}
              />
            )}
          </div>
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
