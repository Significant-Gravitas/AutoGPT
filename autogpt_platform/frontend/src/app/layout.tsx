import React, { Suspense } from "react";
import type { Metadata } from "next";
import { Poppins } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";

import "./globals.css";

import { Toaster } from "@/components/ui/toaster";
import { Providers } from "@/app/providers";
import TallyPopupSimple from "@/components/TallyPopup";
import { GoogleAnalytics } from "@/components/analytics/google-analytics";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-poppins",
});

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
      className={`${poppins.variable} ${GeistSans.variable} ${GeistMono.variable}`}
    >
      <head>
        <GoogleAnalytics
          gaId={process.env.GA_MEASUREMENT_ID || "G-FH2XK2W4GN"} // This is the measurement Id for the Google Analytics dev project
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
          </div>
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
