import React from "react";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Providers } from "@/app/providers";
import { NavBar } from "@/components/NavBar";
import { cn } from "@/lib/utils";

import "./globals.css";
import TallyPopupSimple from "@/components/TallyPopup";
import { GoogleAnalytics } from "@next/third-parties/google";
import { Toaster } from "@/components/ui/toaster";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NextGen AutoGPT",
  description: "Your one stop shop to creating AI Agents",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={cn("antialiased transition-colors", inter.className)}>
        <Providers
          attribute="class"
          defaultTheme="light"
          // Feel free to remove this line if you want to use the system theme by default
          // enableSystem
          disableTransitionOnChange
        >
          <div className="flex min-h-screen flex-col">
            <NavBar />
            <main className="flex-1 overflow-hidden p-4">{children}</main>
            <TallyPopupSimple />
          </div>
          <Toaster />
        </Providers>
      </body>

      <GoogleAnalytics
        gaId={process.env.GA_MEASUREMENT_ID || "G-FH2XK2W4GN"} // This is the measurement Id for the Google Analytics dev project
      />
    </html>
  );
}
