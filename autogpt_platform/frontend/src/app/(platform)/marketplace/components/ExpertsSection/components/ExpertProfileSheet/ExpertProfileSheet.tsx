"use client";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import {
  CaretDownIcon,
  CheckCircleIcon,
  LightningIcon,
} from "@phosphor-icons/react";
import { useState } from "react";
import { getExpertAccent } from "../../helpers";
import { useExpertProfileSheet } from "./useExpertProfileSheet";

interface Props {
  templateId: string | null;
  onClose: () => void;
}

export function ExpertProfileSheet({ templateId, onClose }: Props) {
  const { template, isHired, isHiring, hire } = useExpertProfileSheet(
    templateId,
    onClose,
  );
  const accent = template ? getExpertAccent(template.role) : null;

  return (
    <Dialog
      styling={{ width: "640px" }}
      controlled={{
        isOpen: templateId !== null,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        {template && accent ? (
          <div className="relative">
            <div
              className={cn(
                "relative flex items-center gap-5 overflow-hidden rounded-2xl border border-zinc-200/60 p-5",
                accent.wash,
              )}
            >
              <Avatar className="h-24 w-24 bg-white shadow-sm ring-1 ring-black/5">
                {template.avatar_url ? (
                  <AvatarImage src={template.avatar_url} alt={template.name} />
                ) : null}
                <AvatarFallback>{template.name.slice(0, 2)}</AvatarFallback>
              </Avatar>
              <div>
                <div className="flex items-center gap-3">
                  <h2 className="text-3xl font-semibold tracking-[-0.02em] text-zinc-900">
                    {template.name}
                  </h2>
                  <span
                    className={cn(
                      "rounded-full px-3 py-1 text-sm font-medium",
                      accent.pill,
                    )}
                  >
                    {template.role}
                  </span>
                </div>
                {template.tagline ? (
                  <p className="mt-1.5 text-base text-zinc-500">
                    {template.tagline}
                  </p>
                ) : null}
              </div>
            </div>

            <PersonalitySection
              key={template.id}
              text={template.bio || template.identity}
            />

            {template.workflows.length > 0 ? (
              <div className="relative mt-8">
                <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
                  Preloaded workflows
                </div>
                <div className="divide-y divide-zinc-100 rounded-xl border border-zinc-200/80 bg-white">
                  {template.workflows.map((workflow) => (
                    <div
                      key={workflow.id}
                      className="flex items-center gap-3 px-4 py-3"
                    >
                      <LightningIcon
                        size={18}
                        weight="fill"
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
                  <CheckCircleIcon size={20} weight="fill" />
                  On your team
                </div>
              ) : (
                <Button
                  variant="primary"
                  onClick={hire}
                  loading={isHiring}
                  className="h-12 w-full rounded-full text-base"
                >
                  {`Hire ${template.name}`}
                </Button>
              )}
            </div>
          </div>
        ) : null}
      </Dialog.Content>
    </Dialog>
  );
}

function PersonalitySection({ text }: { text: string }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="relative mt-8">
      <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
        About
      </div>
      <p
        className={cn(
          "whitespace-pre-line text-base leading-relaxed text-zinc-600",
          !isExpanded && "line-clamp-4",
        )}
      >
        {text}
      </p>
      <button
        type="button"
        onClick={() => setIsExpanded((v) => !v)}
        className="mt-2 flex items-center gap-1 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-900"
      >
        {isExpanded ? "Show less" : "Read more"}
        <CaretDownIcon
          size={14}
          weight="bold"
          className={cn(
            "transition-transform duration-200",
            isExpanded && "rotate-180",
          )}
        />
      </button>
    </div>
  );
}
