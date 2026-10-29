import { AuthMarketingPanel } from "@/components/auth/AuthSplitLayout/AuthMarketingPanel";
import {
  BrainIcon,
  SecurityCheckIcon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function SignupMarketingPanel() {
  return (
    <AuthMarketingPanel
      headingLines={[
        "Your ambition",
        <span key="line-2">
          deserves{" "}
          <span
            // TODO(tokens): no exact role for text-violet-600 (#7C3AED)
            className="text-violet-600"
          >
            a team.
          </span>
        </span>,
      ]}
      description="Hire AI experts who research, write, sell and take work off your plate. Otto, your Head of AI, hands out the jobs and reports back to you."
      items={[
        {
          icon: <Icon icon={UserGroupIcon} size={20} />,
          title: "A specialist for every job",
          description:
            "Experts for research, content, sales and more, ready to work in minutes.",
        },
        {
          icon: <Icon icon={BrainIcon} size={20} />,
          title: "They learn your way",
          description:
            "Show them how you like things done once. They remember.",
        },
        {
          icon: <Icon icon={SecurityCheckIcon} size={20} />,
          title: "You stay in charge",
          description:
            "You decide when they check with you first, and their work stays yours.",
        },
      ]}
    />
  );
}
