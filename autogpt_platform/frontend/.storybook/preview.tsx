import React from "react";
import type { Preview } from "@storybook/react";
import { initialize, mswLoader } from "msw-storybook-addon";
import { Inter, Poppins } from "next/font/google";
import localFont from "next/font/local";
import "../src/app/globals.css";
import { Providers } from "../src/app/providers";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-poppins",
});

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

const GeistSans = localFont({
  src: "../fonts/geist-sans/Geist-Variable.woff2",
  variable: "--font-geist-sans",
});

const GeistMono = localFont({
  src: "../fonts/geist-mono/GeistMono-Variable.woff2",
  variable: "--font-geist-mono",
});

// Initialize MSW
initialize();

const preview: Preview = {
  parameters: {
    nextjs: {
      appDirectory: true,
    },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    layout: "fullscreen",
  },
  decorators: [
    (Story, context) => {
      const mockOptions = context.parameters.mockBackend || {};

      return (
        <div
          className={`${poppins.variable} ${inter.variable} ${GeistMono.variable} ${GeistSans.variable}`}
        >
          <Providers mockClientProps={mockOptions}>
            <Story />
          </Providers>
        </div>
      );
    },
  ],
  loaders: [mswLoader],
};

export default preview;
