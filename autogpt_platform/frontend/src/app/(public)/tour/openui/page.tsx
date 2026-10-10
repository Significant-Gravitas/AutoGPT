import type { Metadata } from "next";
import { OpenUILab } from "@/app/(platform)/copilot/openui/components/OpenUILab/OpenUILab";

export const metadata: Metadata = {
  title: "Generative workspace · AutoGPT Labs",
  robots: { index: false, follow: false },
};

export default function OpenUIDemoPage() {
  return <OpenUILab standalone />;
}
