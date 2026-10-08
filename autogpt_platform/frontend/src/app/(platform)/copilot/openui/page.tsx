import { notFound } from "next/navigation";
import { getOpenUIConfig, isOpenUIEnabled } from "@/lib/openui/config";
import { OpenUILab } from "./components/OpenUILab/OpenUILab";

export default function OpenUIPage() {
  if (!isOpenUIEnabled()) notFound();
  return <OpenUILab liveAvailable={Boolean(getOpenUIConfig())} />;
}
