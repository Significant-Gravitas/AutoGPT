import { Expert } from "@/app/api/__generated__/models/expert";
import {
  expertPastel,
  getCategoryHex,
  getExpertTopicHex,
} from "@/components/molecules/ExpertAvatar/colors";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ExpertIdentityDetails } from "@/components/molecules/ExpertIdentityDetails/ExpertIdentityDetails";
import { CategoryTag } from "../../../components/CategoryChip/CategoryTag";

interface BandProps {
  expert: Expert;
}

interface IdentityProps {
  expert: Expert;
  stats: string[];
}

function getTopic(expert: Expert) {
  return getExpertTopicHex({
    avatarUrl: expert.avatar_url,
    categories: expert.categories,
    role: expert.role,
  });
}

const GLOSSY_STAT =
  // TODO(tokens): no exact role: shadow-[…] #101828; no fitting role for shadow-[…] #FFFFFF; #FFFFFF is card (a surface fill), popover (a surface fill), accent-foreground (a text colour)
  "inline-flex h-8 items-center rounded-full border border-border-base bg-gradient-to-b from-surface to-surface-muted px-3.5 text-sm font-medium text-foreground-body shadow-[inset_0_1px_0_rgba(255,255,255,0.9),0_1px_2px_rgba(16,24,40,0.05)]";

/** The hero band: the expert's colour with two soft glows and the category
 *  on it. The portrait (ExpertHeroIdentity) breaks out of its bottom edge,
 *  and the hire card overlaps its right end on wide screens. */
export function ExpertHeroBand({ expert }: BandProps) {
  const area = expert.categories?.[0];
  const topic = getTopic(expert);

  return (
    <div
      className="relative h-44 overflow-hidden rounded-3xl sm:h-56"
      style={{
        backgroundColor: expertPastel(topic),
        backgroundImage: [
          `radial-gradient(circle at 15% 0%, color-mix(in srgb, ${topic} 45%, rgb(var(--color-card))), transparent 55%)`,
          `radial-gradient(circle at 90% 110%, color-mix(in srgb, ${topic} 35%, rgb(var(--color-card))), transparent 50%)`,
        ].join(", "),
      }}
    >
      <div
        aria-hidden
        className="absolute inset-0 bg-[linear-gradient(to_bottom,rgb(var(--color-card)/0.35),transparent_45%)]"
      />
      {area ? (
        <CategoryTag
          category={area}
          size="medium"
          className="absolute left-5 top-5 h-7 px-2.5 text-body-sm leading-inherit"
        />
      ) : null}
    </div>
  );
}

/** The portrait breaking out of the band, the name with its glowing title,
 *  the tagline and what comes with the hire, on the page's left edge. */
export function ExpertHeroIdentity({ expert, stats }: IdentityProps) {
  const area = expert.categories?.[0];
  const topic = getTopic(expert);

  return (
    <header>
      <div className="relative -mt-16 sm:-mt-20">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url}
          color={expert.color}
          backgroundColor={topic}
          size={128}
          // TODO(tokens): no exact role: border-[#e3e3e3] #E3E3E3, shadow-[…] #101828, ring-[#f9f9f9] #F9F9F9
          className="ml-6 rounded-full border border-[#e3e3e3] shadow-[0_12px_32px_-12px_rgba(16,24,40,0.35)] ring-[6px] ring-[#f9f9f9]"
        />
        <div className="mt-4 min-w-0">
          <ExpertIdentityDetails
            name={expert.name}
            role={expert.role}
            jobTitle={expert.job_title}
            size="page"
            nameAlign="baseline"
            areaColor={getCategoryHex(area)}
          />
        </div>
        {expert.tagline?.trim() ? (
          <p className="mt-3 max-w-2xl text-lg leading-8 text-foreground-secondary">
            {expert.tagline}
          </p>
        ) : null}
        {stats.length > 0 ? (
          <ul
            className="mt-5 flex flex-wrap gap-2"
            aria-label="What comes with this expert"
          >
            {stats.map((stat) => (
              <li key={stat} className={GLOSSY_STAT}>
                {stat}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </header>
  );
}
