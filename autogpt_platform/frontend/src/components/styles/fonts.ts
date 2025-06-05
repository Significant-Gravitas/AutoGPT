import { Poppins } from "next/font/google";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";

const poppins = Poppins({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"] as const,
  variable: "--font-poppins",
});

export const fonts = {
  poppins,
  sans: GeistSans,
  mono: GeistMono,
};
