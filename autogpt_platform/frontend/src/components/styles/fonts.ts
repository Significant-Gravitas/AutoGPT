import { Inter, Poppins } from "next/font/google";
import { GeistSans as geistSans } from "geist/font/sans";
import { GeistMono as geistMono } from "geist/font/mono";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"] as const,
  variable: "--font-poppins",
  display: "swap",
  preload: true,
});

// Variable axis so the Linear-style 510/590 weights resolve exactly.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
  preload: false,
});

export const fonts = {
  poppins,
  inter,
  sans: geistSans,
  mono: geistMono,
};
