"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { LockIcon, PencilEdit02Icon } from "@hugeicons/core-free-icons";

interface Props {
  expert: Expert;
  onEditSoul: () => void;
  onFire: () => void;
}

export function ExpertSettingsSection({ expert, onEditSoul, onFire }: Props) {
  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-zinc-200/80 bg-white p-5">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <Text variant="large-medium">Soul</Text>
            <Text variant="small" className="text-zinc-500">
              A living document that shapes every reply.
            </Text>
          </div>
          <Button
            variant="secondary"
            size="small"
            leftIcon={<Icon icon={PencilEdit02Icon} size={16} />}
            onClick={onEditSoul}
          >
            Edit Soul
          </Button>
        </div>
        <dl className="space-y-4">
          <SoulEntry label="Identity" value={expert.identity} />
          <SoulEntry label="Voice" value={expert.voice_preferences} />
          <SoulEntry label="Boundaries" value={expert.boundaries} />
        </dl>
      </section>

      <section className="rounded-2xl border border-zinc-200/80 bg-white p-5">
        <div className="mb-3 flex items-center gap-2">
          <Icon icon={LockIcon} size={16} className="text-zinc-500" />
          <Text variant="large-medium">Protected rules</Text>
        </div>
        <div className="space-y-2 rounded-xl bg-zinc-50 p-4">
          {expert.protected_soul_rules.map((rule) => (
            <div
              key={rule}
              className="flex gap-2 text-sm leading-5 text-zinc-600"
            >
              <Icon icon={LockIcon} size={14} className="mt-0.5 shrink-0" />
              <span>{rule}</span>
            </div>
          ))}
        </div>
        <Text variant="small" className="mt-3 text-zinc-400">
          These safeguards are always active and cannot be edited.
        </Text>
      </section>

      <section className="rounded-2xl border border-red-200 bg-red-50/50 p-5">
        <Text variant="large-medium" className="text-red-700">
          Danger zone
        </Text>
        <Text variant="small" className="mt-1 text-red-600">
          Firing {expert.name} pauses every schedule and removes them from your
          team.
        </Text>
        <Button
          variant="destructive"
          size="small"
          className="mt-4"
          onClick={onFire}
          data-testid="expert-fire-button"
        >
          Fire {expert.name}
        </Button>
      </section>
    </div>
  );
}

function SoulEntry({ label, value }: { label: string; value: string | null }) {
  return (
    <div>
      <dt className="text-xs font-medium uppercase tracking-[0.12em] text-purple-600">
        {label}
      </dt>
      <dd className="mt-1 whitespace-pre-line text-sm leading-relaxed text-zinc-600">
        {value || "Not set yet."}
      </dd>
    </div>
  );
}
