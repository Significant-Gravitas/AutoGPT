import { AuthMarketingPanel } from "@/components/auth/AuthSplitLayout/AuthMarketingPanel";
import { LoginWhatsNew } from "./LoginWhatsNew/LoginWhatsNew";

export function LoginMarketingPanel() {
  return (
    <AuthMarketingPanel
      headingLines={["Welcome back"]}
      description="Pick up where you left off. Your experts are waiting."
    >
      <LoginWhatsNew />
    </AuthMarketingPanel>
  );
}
