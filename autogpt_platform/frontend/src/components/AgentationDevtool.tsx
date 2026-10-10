"use client";

import dynamic from "next/dynamic";
import { useNativeApp } from "@/hooks/useNativeApp";

const Agentation = dynamic(
  () => import("agentation").then((mod) => mod.Agentation),
  { ssr: false },
);

export default function AgentationDevtool() {
  const isNativeApp = useNativeApp();
  return isNativeApp ? null : <Agentation />;
}
