"use client";

import { notFound } from "next/navigation";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ToolUIPreview } from "./components/ToolUIPreview/ToolUIPreview";

export default function ToolUIPreviewPage() {
  const isNewToolUI = useGetFlag(Flag.NEW_TOOL_UI);

  if (!isNewToolUI) {
    notFound();
  }

  return <ToolUIPreview />;
}
