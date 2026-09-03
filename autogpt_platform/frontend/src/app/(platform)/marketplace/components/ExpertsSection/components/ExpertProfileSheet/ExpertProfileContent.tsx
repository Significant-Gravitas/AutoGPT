"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { ExpertAccent } from "../../helpers";
import {
  ArrowDown01Icon,
  CheckmarkCircle02Icon,
  FlashIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  expert: Expert;
  accent: ExpertAccent;
  isHired: boolean;
  isHiring: boolean;
  onHire: () => void;
}

export function ExpertProfileContent({
  expert,
  accent,
  isHired,
  isHiring,
  onHire,
}: Props) {
  return (
    <div className="relative">
      <div
        className={cn(
          "relative flex items-center gap-5 overflow-hidden rounded-2xl border border-zinc-200/60 p-5",
          accent.wash,
        )}
      >
        <Avatar className="h-24 w-24 bg-white shadow-sm ring-1 ring-black/5">
          {expert.avatar_url ? (
            <AvatarImage src={expert.avatar_url} alt={expert.name} />
          ) : null}
          <AvatarFallback>{expert.name}</AvatarFallback>
        </Avatar>
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-3xl font-semibold tracking-[-0.02em] text-zinc-900">
              {expert.name}
            </h2>
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium",
                accent.pill,
              )}
            >
              <Icon icon={accent.roleIcon} size={14} />
              {expert.role}
            </span>
          </div>
          {expert.tagline ? (
            <p className="mt-1.5 text-base text-zinc-500">{expert.tagline}</p>
          ) : null}
        </div>
      </div>

      <PersonalitySection
        key={expert.id}
        text={expert.bio || expert.identity}
      />

      {expert.skills && expert.skills.length > 0 ? (
        <div className="relative mt-8">
          <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
            Skills
          </div>
          <div className="flex flex-wrap gap-2">
            {expert.skills.map((skill) => (
              <span
                key={skill}
                className="rounded-full bg-zinc-50 px-3 py-1.5 text-sm text-zinc-600 ring-1 ring-inset ring-zinc-200/80"
              >
                {skill}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {expert.workflows.length > 0 ? (
        <div className="relative mt-8">
          <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
            {expert.is_template ? "Preloaded workflows" : "Workflows"}
          </div>
          <div className="divide-y divide-zinc-100 rounded-xl border border-zinc-200/80 bg-white">
            {expert.workflows.map((workflow) => (
              <div
                key={workflow.id}
                className="flex items-center gap-3 px-4 py-3"
              >
                <Icon
                  icon={FlashIcon}
                  size={18}
                  className={cn("shrink-0", accent.icon)}
                />
                <div className="min-w-0">
                  <div className="text-[15px] font-medium text-zinc-800">
                    {workflow.name ?? "Unnamed workflow"}
                  </div>
                  {workflow.description ? (
                    <div className="line-clamp-1 text-[13px] text-zinc-500">
                      {workflow.description}
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="relative mt-8">
        {isHired ? (
          <div className="flex h-12 w-full items-center justify-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 text-base font-medium text-emerald-700">
            <Icon icon={CheckmarkCircle02Icon} size={20} />
            On your team
          </div>
        ) : (
          <Button
            variant="primary"
            onClick={onHire}
            loading={isHiring}
            className="h-12 w-full rounded-full text-base"
          >
            {`Hire ${expert.name}`}
          </Button>
        )}
      </div>
    </div>
  );
}

// Roughly the number of characters that fit in the four-line clamp below.
const CLAMPED_BIO_LENGTH = 280;

function PersonalitySection({ text }: { text: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const isClampable = text.length > CLAMPED_BIO_LENGTH;

  return (
    <div className="relative mt-8">
      <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
        About
      </div>
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
          onClick={() => setIsExpanded((v) => !v)}
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
