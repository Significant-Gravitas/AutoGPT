import type { ReactNode } from "react";
import { NewChatProvider } from "./NewChatContext";
import { CopilotShell } from "./components/CopilotShell/CopilotShell";

export default function CopilotLayout({ children }: { children: ReactNode }) {
  return (
    <NewChatProvider>
      <CopilotShell>{children}</CopilotShell>
    </NewChatProvider>
  );
}
