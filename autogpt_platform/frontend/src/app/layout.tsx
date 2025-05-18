import React, { Suspense } from "react";
import type { Metadata } from "next";
import { Inter, Poppins } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";

import { cn } from "@/lib/utils";
import "./globals.css";

import { Toaster } from "@/components/ui/toaster";
import { Providers } from "@/app/providers";
import TallyPopupSimple from "@/components/TallyPopup";
import OttoChatWidget from "@/components/OttoChatWidget";
import { GoogleAnalytics } from "@/components/analytics/google-analytics";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-poppins",
});

export const metadata: Metadata = {
  title: "NextGen AutoGPT",
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
      className={`${poppins.variable} ${GeistSans.variable} ${GeistMono.variable} ${inter.variable}`}
    >
      <head>
        <GoogleAnalytics
          gaId={process.env.GA_MEASUREMENT_ID || "G-FH2XK2W4GN"} // This is the measurement Id for the Google Analytics dev project
        />
      </head>
      <body
        className={cn(
          "bg-neutral-50 antialiased transition-colors",
          inter.className,
        )}
      >
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
            <Suspense fallback={null}>
              <OttoChatWidget />
            </Suspense>
          </div>
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
