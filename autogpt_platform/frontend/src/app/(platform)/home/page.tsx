"use client";

import { CopilotPage } from "../copilot/CopilotPage";
import { useCompactModeRedirect } from "../copilot/compact/useCompactModeRedirect";

export default function HomePage() {
  const { isRedirecting } = useCompactModeRedirect();
  if (isRedirecting) return null;
  return <CopilotPage />;
}
