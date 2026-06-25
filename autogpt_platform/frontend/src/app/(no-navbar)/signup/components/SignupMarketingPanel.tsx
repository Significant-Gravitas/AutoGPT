import {
  HammerIcon,
  LightningIcon,
  ShieldCheckIcon,
} from "@phosphor-icons/react/dist/ssr";
import { AuthMarketingPanel } from "@/components/auth/AuthSplitLayout/AuthMarketingPanel";

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
          icon: <LightningIcon size={20} weight="duotone" />,
          title: "Run in minutes",
          description: "Get started quickly and see results fast.",
        },
        {
          icon: <HammerIcon size={20} weight="duotone" />,
          title: "Built for real work",
          description: "Powerful blocks that handle tasks across your stack.",
        },
        {
          icon: <ShieldCheckIcon size={20} weight="duotone" />,
          title: "Secure & private",
          description: "Enterprise-grade security to keep your data safe.",
        },
      ]}
    />
  );
}
