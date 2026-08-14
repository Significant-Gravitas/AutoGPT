"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon, FlashIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { ExpertAccent } from "../../../../helpers";
import { ProfileSection } from "../ProfileSection/ProfileSection";

const CLAMPED_BIO_LENGTH = 280;

type Props = {
  firstName: string;
  firstWorkflow: ExpertWorkflowRef | null;
  bio: string | null;
  accent: ExpertAccent;
};

export function DayOneSection({
  firstName,
  firstWorkflow,
  bio,
  accent,
}: Props) {
  const trimmedBio = bio?.trim() || null;
  if (!firstWorkflow && !trimmedBio) return null;

  return (
    <ProfileSection title={`What ${firstName} sets up on day one`}>
      {firstWorkflow ? (
        <div className="mb-3 flex items-start gap-3 rounded-xl border border-zinc-200/80 bg-white px-4 py-3">
          <Icon
            icon={FlashIcon}
            size={18}
            className={cn("mt-0.5 shrink-0", accent.icon)}
          />
          <div className="min-w-0">
            <div className="text-[15px] font-medium text-zinc-800">
              {firstWorkflow.name}
            </div>
            {firstWorkflow.description ? (
              <div className="text-[13px] leading-relaxed text-zinc-500">
                {firstWorkflow.description}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      {trimmedBio ? <BioText text={trimmedBio} /> : null}
    </ProfileSection>
  );
}

function BioText({ text }: { text: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isClampable = text.length > CLAMPED_BIO_LENGTH;

  return (
    <div>
      <p
        className={cn(
          "whitespace-pre-line text-base leading-relaxed text-zinc-600",
          isClampable && !isExpanded && "line-clamp-4",
        )}
      >
        {text}
      </p>
      {isClampable ? (
        <button
          type="button"
          onClick={() => setIsExpanded((isOpen) => !isOpen)}
          className="mt-2 flex items-center gap-1 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-900"
        >
          {isExpanded ? "Show less" : "Read more"}
          <Icon
            icon={ArrowDown01Icon}
            size={14}
            className={cn(
              "transition-transform duration-200",
              isExpanded && "rotate-180",
            )}
          />
        </button>
      ) : null}
    </div>
  );
}
