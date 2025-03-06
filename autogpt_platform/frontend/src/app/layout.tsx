import React from "react";
import type { Metadata } from "next";
import { Inter, Poppins } from "next/font/google";
import { GoogleAnalytics } from "@next/third-parties/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { headers } from "next/headers";

import { cn } from "@/lib/utils";
import "./globals.css";

import { Navbar } from "@/components/agptui/Navbar";
import { Toaster } from "@/components/ui/toaster";
import { IconType } from "@/components/ui/icons";
import { Providers } from "@/app/providers";
import TallyPopupSimple from "@/components/TallyPopup";
import OttoChatWidget from "@/components/OttoChatWidget";

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
  const pathname = headers().get("x-current-path");
  const isOnboarding = pathname?.startsWith("/onboarding");

  return (
    <html
      lang="en"
      className={`${poppins.variable} ${GeistSans.variable} ${GeistMono.variable} ${inter.variable}`}
    >
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
            {!isOnboarding && (
              <Navbar
                links={[
                  {
                    name: "Marketplace",
                    href: "/marketplace",
                  },
                  {
                    name: "Library",
                    href: "/library",
                  },
                  {
                    name: "Build",
                    href: "/build",
                  },
                ]}
                menuItemGroups={[
                  {
                    items: [
                      {
                        icon: IconType.Edit,
                        text: "Edit profile",
                        href: "/profile",
                      },
                    ],
                  },
                  {
                    items: [
                      {
                        icon: IconType.LayoutDashboard,
                        text: "Creator Dashboard",
                        href: "/profile/dashboard",
                      },
                      {
                        icon: IconType.UploadCloud,
                        text: "Publish an agent",
                      },
                    ],
                  },
                  {
                    items: [
                      {
                        icon: IconType.Settings,
                        text: "Settings",
                        href: "/profile/settings",
                      },
                    ],
                  },
                  {
                    items: [
                      {
                        icon: IconType.LogOut,
                        text: "Log out",
                      },
                    ],
                  },
                ]}
              />
            )}
            <main className="w-full flex-grow">{children}</main>
            <TallyPopupSimple />
            <OttoChatWidget />
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
