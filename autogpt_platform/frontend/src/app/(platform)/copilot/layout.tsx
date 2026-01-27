import type { ReactNode } from "react";
import { CopilotShell } from "./components/CopilotShell/CopilotShell";

export default function CopilotLayout({ children }: { children: ReactNode }) {
  return <CopilotShell>{children}</CopilotShell>;
}
