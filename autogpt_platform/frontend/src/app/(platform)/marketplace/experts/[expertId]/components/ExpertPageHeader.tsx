import { getExpertTopicHex } from "@/components/molecules/ExpertAvatar/colors";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import { ExpertTagline } from "@/components/molecules/ExpertIdentityDetails/components/ExpertTagline";
import { ReactNode } from "react";
import { CategoryTag } from "../../../components/CategoryChip/CategoryTag";

interface Props {
  expert: Expert;
  actions: ReactNode;
}

export function ExpertPageHeader({ expert, actions }: Props) {
  const area = expert.categories?.[0];

  return (
    <header>
      <div className="grid grid-cols-[96px_minmax(0,1fr)] items-center gap-4 sm:gap-x-5">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url}
          color={expert.color}
          backgroundColor={getExpertTopicHex({
            avatarUrl: expert.avatar_url,
            categories: expert.categories,
            role: expert.role,
          })}
          size={96}
          className="border border-black/5"
        />
        <div className="min-w-0 flex-1">
          <ExpertIdentityDetails
            name={expert.name}
            role={expert.role}
            jobTitle={expert.job_title}
            size="page"
            nameAlign="baseline"
          />
          {area ? (
            <CategoryTag category={area} size="default" className="mt-2" />
          ) : null}
        </div>
        <div className="col-span-2 min-w-0 sm:col-span-1 sm:col-start-2">
          {actions}
        </div>
      </div>
      <ExpertTagline tagline={expert.tagline} />
    </header>
  );
}
