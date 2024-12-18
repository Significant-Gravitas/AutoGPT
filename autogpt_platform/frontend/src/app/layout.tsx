import React from "react";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Providers } from "@/app/providers";
import { cn } from "@/lib/utils";
import { Navbar } from "@/components/agptui/Navbar";

import "./globals.css";
import TallyPopupSimple from "@/components/TallyPopup";
import { GoogleAnalytics } from "@next/third-parties/google";
import { Toaster } from "@/components/ui/toaster";
import { IconType } from "@/components/ui/icons";

const inter = Inter({ subsets: ["latin"] });

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
    <html lang="en">
      <body className={cn("antialiased transition-colors", inter.className)}>
        <Providers
          attribute="class"
          defaultTheme="light"
          // Feel free to remove this line if you want to use the system theme by default
          // enableSystem
          disableTransitionOnChange
        >
          <div className="flex min-h-screen flex-col items-center justify-center">
            <Navbar
              links={[
                {
                  name: "Agent Store",
                  href: "/store",
                },
                {
                  name: "Library",
                  href: "/monitoring",
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
                      href: "/store/profile",
                    },
                  ],
                },
                {
                  items: [
                    {
                      icon: IconType.LayoutDashboard,
                      text: "Creator Dashboard",
                      href: "/store/dashboard",
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
                      href: "/store/settings",
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
            <main className="flex-1 p-4">{children}</main>
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
