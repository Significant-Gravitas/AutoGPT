import React from 'react';
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { Providers } from "@/app/providers";
import {NavBar} from "@/components/NavBar";
import {cn} from "@/lib/utils";

import "./globals.css";

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
        <body className={
            cn(
                'antialiased transition-colors',
                inter.className
            )
        }>
        <Providers
            attribute="class"
            defaultTheme="light"
            // Feel free to remove this line if you want to use the system theme by default
            // enableSystem
            disableTransitionOnChange
        >
            <div className="flex flex-col min-h-screen ">
                <NavBar/>
                <main className="flex-1 p-4 overflow-hidden">
                    {children}
                </main>
            </div>
        </Providers>
        </body>
        </html>
    );
}
