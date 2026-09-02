import type { Metadata } from "next";
import { TourHero } from "./components/TourHero/TourHero";

const TITLE = "Watch AutoGPT build a working agent in 60 seconds";
const DESCRIPTION =
  "See AutoGPT build and run a real AI agent from start to finish in about 60 seconds. No signup required. Watch how AI agents automate real work before you try it yourself.";
const OG_IMAGE = "https://platform.agpt.co/images/tour-og.png";

export const metadata: Metadata = {
  title: TITLE,
  description: DESCRIPTION,
  openGraph: {
    title: TITLE,
    description: DESCRIPTION,
    siteName: "AutoGPT",
    type: "website",
    images: [
      {
        url: OG_IMAGE,
        width: 1200,
        height: 630,
        alt: "AutoGPT tour — an agent built, run and delivering its first result",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
    images: [OG_IMAGE],
  },
};

export default function Page() {
  return <TourHero />;
}
