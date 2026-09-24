import { getExpertTopicHex } from "@/components/molecules/ExpertAvatar/colors";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import { ExpertTagline } from "@/components/molecules/ExpertIdentityDetails/components/ExpertTagline";
import { ReactNode } from "react";

interface Props {
  expert: Expert;
  actions: ReactNode;
}

export function ExpertPageHeader({ expert, actions }: Props) {
  return (
    <header>
      <div className="flex flex-wrap items-center gap-4 sm:gap-5">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url}
          color={expert.color}
          backgroundColor={getExpertTopicHex(expert.role, expert.categories)}
          size={96}
        />
        <div className="min-w-0 flex-1">
          <ExpertIdentityDetails
            name={expert.name}
            role={expert.role}
            jobTitle={expert.job_title}
            size="page"
          />
        </div>
        <div className="w-full sm:w-auto">{actions}</div>
      </div>
      <ExpertTagline tagline={expert.tagline} />
    </header>
  );
}
