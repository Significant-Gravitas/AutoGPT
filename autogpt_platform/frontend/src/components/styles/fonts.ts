import { Poppins } from "next/font/google";
import { GeistSans as geistSans } from "geist/font/sans";
import { GeistMono as geistMono } from "geist/font/mono";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"] as const,
  variable: "--font-poppins",
  display: "swap",
  preload: true,
});

export const fonts = {
  poppins,
  sans: geistSans,
  mono: geistMono,
};
