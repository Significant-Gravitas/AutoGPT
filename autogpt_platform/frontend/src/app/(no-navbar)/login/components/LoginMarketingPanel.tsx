import { AuthMarketingPanel } from "@/components/auth/AuthSplitLayout/AuthMarketingPanel";
import { BrainIcon, GaugeIcon, Store01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function LoginMarketingPanel() {
  return (
    <AuthMarketingPanel
      headingLines={["Welcome back"]}
      description="Pick up where you left off. Your agents are waiting."
      itemsTitle="What's new"
      items={[
        {
          icon: <Icon icon={BrainIcon} size={20} />,
          title: "New memory upgrades",
          description: "Smarter agents with longer context windows.",
        },
        {
          icon: <Icon icon={Store01Icon} size={20} />,
          title: "Marketplace update",
          description: "Discover agents shared by the community.",
        },
        {
          icon: <Icon icon={GaugeIcon} size={20} />,
          title: "Performance boost",
          description: "Faster runs and reduced latency.",
        },
      ]}
    />
  );
}
