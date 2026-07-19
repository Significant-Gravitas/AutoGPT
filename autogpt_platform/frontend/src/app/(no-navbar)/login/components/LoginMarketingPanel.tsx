import {
  BrainIcon,
  GaugeIcon,
  StorefrontIcon,
} from "@/components/atoms/Icon/phosphor";
import { AuthMarketingPanel } from "@/components/auth/AuthSplitLayout/AuthMarketingPanel";

export function LoginMarketingPanel() {
  return (
    <AuthMarketingPanel
      headingLines={["Welcome back"]}
      description="Pick up where you left off. Your agents are waiting."
      itemsTitle="What's new"
      items={[
        {
          icon: <BrainIcon size={20} weight="duotone" />,
          title: "New memory upgrades",
          description: "Smarter agents with longer context windows.",
        },
        {
          icon: <StorefrontIcon size={20} weight="duotone" />,
          title: "Marketplace update",
          description: "Discover agents shared by the community.",
        },
        {
          icon: <GaugeIcon size={20} weight="duotone" />,
          title: "Performance boost",
          description: "Faster runs and reduced latency.",
        },
      ]}
    />
  );
}
