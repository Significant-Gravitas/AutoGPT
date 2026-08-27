import { AuthMarketingPanel } from "@/components/auth/AuthSplitLayout/AuthMarketingPanel";
import {
  FlashIcon,
  HammerIcon,
  SecurityCheckIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function SignupMarketingPanel() {
  return (
    <AuthMarketingPanel
      headingLines={[
        "AI agents",
        <span key="line-2">
          that work <span className="text-slate-400">for you.</span>
        </span>,
      ]}
      description="Discover, build, and deploy AI agents that automate real work — no code required. Start building agents in minutes."
      items={[
        {
          icon: <Icon icon={FlashIcon} size={20} />,
          title: "Run in minutes",
          description: "Get started quickly and see results fast.",
        },
        {
          icon: <Icon icon={HammerIcon} size={20} />,
          title: "Built for real work",
          description: "Powerful blocks that handle tasks across your stack.",
        },
        {
          icon: <Icon icon={SecurityCheckIcon} size={20} />,
          title: "Secure & private",
          description: "Enterprise-grade security to keep your data safe.",
        },
      ]}
    />
  );
}
