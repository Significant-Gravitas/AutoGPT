"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertSoulUpdate } from "@/app/api/__generated__/models/expertSoulUpdate";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { LockIcon } from "@hugeicons/core-free-icons";
import { ReactNode, useState } from "react";
import { cn } from "@/lib/utils";
import { ExpertSidePanel } from "../ExpertSidePanel/ExpertSidePanel";
import { useBottomScrollShadow } from "./useBottomScrollShadow";
import { useSoulDrawer } from "./useSoulDrawer";

interface Props {
  expert: Expert | null;
  onClose: () => void;
}

export function SoulDrawer({ expert, onClose }: Props) {
  return (
    <ExpertSidePanel
      identity={
        expert ? { name: expert.name, avatarUrl: expert.avatar_url } : null
      }
      title={expert ? `${expert.name}'s Soul` : ""}
      panelId="soul"
      closeLabel="Close Soul panel"
      onClose={onClose}
    >
      {expert ? <SoulPanelBody expert={expert} onClose={onClose} /> : null}
    </ExpertSidePanel>
  );
}

interface BodyProps {
  expert: Expert;
  onClose: () => void;
}

function SoulPanelBody({ expert, onClose }: BodyProps) {
  const { soul, updateField, save, isPending, canSave } = useSoulDrawer({
    expert,
    onClose,
  });
  const [scrollElement, setScrollElement] = useState<HTMLDivElement | null>(
    null,
  );
  const hasMoreBelow = useBottomScrollShadow(scrollElement);

  return (
    <form onSubmit={save} className="flex min-h-0 flex-1 flex-col">
      <div className="relative min-h-0 flex-1">
        <div
          ref={setScrollElement}
          className="h-full overflow-y-auto px-5 py-5"
        >
          <Text variant="small" tone="muted" className="mb-5">
            A living document that shapes every reply.
          </Text>
          <SoulFields soul={soul} updateField={updateField} />
          <LearnedNotes />
          <ProtectedRules rules={expert.protected_soul_rules} />
        </div>
        <div
          aria-hidden="true"
          className={cn(
            "pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-sidebar to-transparent transition-opacity duration-200",
            hasMoreBelow ? "opacity-100" : "opacity-0",
          )}
        />
      </div>
      <div className="flex shrink-0 justify-end gap-2 border-t border-t-sidebar-border px-5 py-3">
        <Button type="button" variant="ghost" size="xs" onClick={onClose}>
          Cancel
        </Button>
        <Button
          type="submit"
          variant="primary"
          size="xs"
          loading={isPending}
          disabled={!canSave}
        >
          Save Soul
        </Button>
      </div>
    </form>
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
        labelClassName="!text-zinc-700"
        value={soul.name}
        maxLength={100}
        required
        onChange={(event) => updateField("name", event.target.value)}
      />
      <Input
        id="soul-identity"
        label="Identity and personality"
        labelVariant="small-medium"
        labelClassName="!text-zinc-700"
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
        labelClassName="!text-zinc-700"
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
        labelClassName="!text-zinc-700"
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
      <Text variant="small" tone="muted">
        Nothing recorded yet. What this expert learns will appear here.
      </Text>
    </section>
  );
}

function ProtectedRules({ rules }: { rules: string[] }) {
  return (
    <section className="border-t border-zinc-200 pt-6">
      <div className="mb-3 flex items-center gap-2">
        <Icon icon={LockIcon} size={16} className="text-zinc-500" />
        <SoulSectionTitle>Protected rules</SoulSectionTitle>
      </div>
      <div className="space-y-2 rounded-xl bg-zinc-50 p-4">
        {rules.map((rule) => (
          <Text
            key={rule}
            variant="body"
            as="div"
            tone="secondary"
            className="flex gap-2 leading-5"
          >
            <Icon icon={LockIcon} size={14} className="mt-0.5 shrink-0" />
            <span>{rule}</span>
          </Text>
        ))}
      </div>
      <Text variant="small" tone="muted" className="mt-3">
        These rules are part of every expert&apos;s soul and cannot be edited.
      </Text>
    </section>
  );
}

function SoulSectionTitle({ children }: { children: ReactNode }) {
  return (
    <Text variant="body-medium" as="h3" tone="primary">
      {children}
    </Text>
  );
}
