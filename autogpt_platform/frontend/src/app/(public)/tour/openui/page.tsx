import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { isOpenUIEnabled } from "@/lib/openui/config";
import { OpenUILab } from "@/app/(platform)/copilot/openui/components/OpenUILab/OpenUILab";

export const metadata: Metadata = {
  title: "Generative workspace · AutoGPT Labs",
  robots: { index: false, follow: false },
};

export default function OpenUIDemoPage() {
  if (!isOpenUIEnabled()) notFound();
  return <OpenUILab standalone liveAvailable={false} />;
}
