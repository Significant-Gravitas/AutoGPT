"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertSoulUpdate } from "@/app/api/__generated__/models/expertSoulUpdate";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
} from "@/components/ui/sheet";
import { LockSimpleIcon } from "@phosphor-icons/react";
import { ReactNode, useState } from "react";
import { useSoulDrawer } from "./useSoulDrawer";

interface Props {
  expert: Expert | null;
  onClose: () => void;
}

export function SoulDrawer({ expert, onClose }: Props) {
  const [displayedExpert] = useState(expert);
  const { soul, updateField, save, isPending, canSave } = useSoulDrawer({
    expert,
    onClose,
  });

  return (
    <Sheet
      open={expert !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <SheetContent
        side="right"
        className="flex w-full flex-col overflow-hidden p-0 sm:w-1/2 sm:max-w-none"
      >
        <div className="border-b border-zinc-200 bg-white px-6 py-5 pr-12 sm:px-8">
          <div className="flex items-center gap-3">
            <Avatar className="h-11 w-11">
              {displayedExpert?.avatar_url ? (
                <AvatarImage
                  src={displayedExpert.avatar_url}
                  alt={displayedExpert.name}
                />
              ) : null}
              <AvatarFallback>{displayedExpert?.name ?? "Soul"}</AvatarFallback>
            </Avatar>
            <div>
              <SheetTitle>{`${displayedExpert?.name ?? "Expert"}'s Soul`}</SheetTitle>
              <SheetDescription>
                A living document that shapes every reply.
              </SheetDescription>
            </div>
          </div>
        </div>

        <form onSubmit={save} className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 overflow-y-auto bg-zinc-50 px-4 py-6 sm:px-8">
            <div className="mx-auto max-w-2xl rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm sm:p-8">
              <SoulFields soul={soul} updateField={updateField} />
              <LearnedNotes />
              <ProtectedRules
                rules={displayedExpert?.protected_soul_rules ?? []}
              />
            </div>
          </div>
          <div className="flex justify-end gap-2 border-t border-zinc-200 bg-white px-6 py-4 sm:px-8">
            <Button type="button" variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button
              type="submit"
              variant="primary"
              loading={isPending}
              disabled={!canSave}
            >
              Save Soul
            </Button>
          </div>
        </form>
      </SheetContent>
    </Sheet>
  );
}

interface SoulFieldsProps {
  soul: ExpertSoulUpdate;
  updateField: (field: keyof ExpertSoulUpdate, value: string) => void;
}

function SoulFields({ soul, updateField }: SoulFieldsProps) {
  return (
    <div className="space-y-1">
      <Input
        id="soul-name"
        label="Name"
        labelVariant="small-medium"
        labelClassName="uppercase tracking-[0.12em] !text-purple-600"
        value={soul.name}
        maxLength={100}
        required
        onChange={(event) => updateField("name", event.target.value)}
      />
      <Input
        id="soul-identity"
        label="Identity and personality"
        labelVariant="small-medium"
        labelClassName="uppercase tracking-[0.12em] !text-purple-600"
        type="textarea"
        rows={6}
        value={soul.identity}
        maxLength={10000}
        required
        onChange={(event) => updateField("identity", event.target.value)}
      />
      <Input
        id="soul-voice"
        label="Voice"
        labelVariant="small-medium"
        labelClassName="uppercase tracking-[0.12em] !text-purple-600"
        type="textarea"
        rows={3}
        value={soul.voice_preferences}
        maxLength={4000}
        placeholder="How should this expert sound?"
        onChange={(event) =>
          updateField("voice_preferences", event.target.value)
        }
      />
      <Input
        id="soul-boundaries"
        label="Boundaries"
        labelVariant="small-medium"
        labelClassName="uppercase tracking-[0.12em] !text-purple-600"
        type="textarea"
        rows={4}
        value={soul.boundaries}
        maxLength={4000}
        placeholder="What should this expert avoid or ask about first?"
        onChange={(event) => updateField("boundaries", event.target.value)}
      />
    </div>
  );
}

function LearnedNotes() {
  return (
    <section className="mb-8">
      <SoulSectionTitle>What I&apos;ve learned</SoulSectionTitle>
      <Text variant="small" className="text-zinc-500">
        Nothing recorded yet. What this expert learns will appear here.
      </Text>
    </section>
  );
}

function ProtectedRules({ rules }: { rules: string[] }) {
  return (
    <section className="border-t border-zinc-200 pt-6">
      <div className="mb-3 flex items-center gap-2">
        <LockSimpleIcon size={16} weight="fill" className="text-zinc-500" />
        <SoulSectionTitle>Protected rules</SoulSectionTitle>
      </div>
      <div className="space-y-2 rounded-xl bg-zinc-50 p-4">
        {rules.map((rule) => (
          <div
            key={rule}
            className="flex gap-2 text-sm leading-5 text-zinc-600"
          >
            <LockSimpleIcon size={14} className="mt-0.5 shrink-0" />
            <span>{rule}</span>
          </div>
        ))}
      </div>
      <Text variant="small" className="mt-3 text-zinc-400">
        These safeguards are always active and cannot be edited.
      </Text>
    </section>
  );
}

function SoulSectionTitle({ children }: { children: ReactNode }) {
  return (
    <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-purple-600">
      {children}
    </h3>
  );
}
