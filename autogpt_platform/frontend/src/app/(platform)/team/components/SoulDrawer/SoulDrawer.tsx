"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertSoulUpdate } from "@/app/api/__generated__/models/expertSoulUpdate";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { FullscreenDialog } from "@/components/molecules/FullscreenDialog/FullscreenDialog";
import { Text } from "@/components/atoms/Text/Text";
import { Cancel01Icon, LockIcon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ReactNode, useEffect, useState } from "react";
import { PanelResizeHandle } from "@/app/(platform)/copilot/components/PanelResizeHandle";
import { useIsMobile } from "@/app/(platform)/copilot/useIsMobile";
import { cn } from "@/lib/utils";
import { useBottomScrollShadow } from "./useBottomScrollShadow";
import { useSoulDrawer } from "./useSoulDrawer";
import { useSoulPanelSidebarCollapse } from "./useSoulPanelSidebarCollapse";

const DEFAULT_PANEL_WIDTH = 520;
const MIN_PANEL_WIDTH = 320;
const MAX_PANEL_VIEWPORT_RATIO = 0.4;
const PANEL_EASE: [number, number, number, number] = [0.32, 0.72, 0, 1];
const PANEL_DURATION = 0.3;

interface Props {
  expert: Expert | null;
  onClose: () => void;
}

export function SoulDrawer({ expert, onClose }: Props) {
  const isMobile = useIsMobile();
  const shouldReduceMotion = useReducedMotion();
  const [width, setWidth] = useState(DEFAULT_PANEL_WIDTH);
  const [isResizing, setIsResizing] = useState(false);
  const maxWidth = useMaxPanelWidth();
  const renderedWidth = Math.min(width, maxWidth);
  useSoulPanelSidebarCollapse(expert !== null);

  if (isMobile) {
    if (!expert) return null;
    return (
      <FullscreenDialog title={`${expert.name}'s Soul`} onClose={onClose}>
        <SoulPanelBody expert={expert} onClose={onClose} />
      </FullscreenDialog>
    );
  }

  const transition =
    shouldReduceMotion || isResizing
      ? { duration: 0 }
      : { duration: PANEL_DURATION, ease: PANEL_EASE };

  return (
    <AnimatePresence>
      {expert ? (
        <motion.aside
          data-soul-panel
          aria-label={`${expert.name}'s Soul`}
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: renderedWidth, opacity: 1 }}
          exit={{ width: 0, opacity: 0 }}
          transition={transition}
          className="sticky top-0 h-svh shrink-0 self-start border-l border-l-[#80808017] bg-sidebar"
        >
          <PanelResizeHandle
            panelSelector="[data-soul-panel]"
            onWidthChange={setWidth}
            onResizingChange={setIsResizing}
            minWidth={MIN_PANEL_WIDTH}
            maxWidth={maxWidth}
          />
          <div className="h-full overflow-hidden">
            <div
              style={{ width: renderedWidth }}
              className="flex h-full min-h-0 flex-col"
            >
              <SoulPanelBody expert={expert} onClose={onClose} />
            </div>
          </div>
        </motion.aside>
      ) : null}
    </AnimatePresence>
  );
}

function useMaxPanelWidth() {
  const [maxWidth, setMaxWidth] = useState(DEFAULT_PANEL_WIDTH);

  useEffect(() => {
    function update() {
      setMaxWidth(
        Math.max(
          MIN_PANEL_WIDTH,
          Math.round(window.innerWidth * MAX_PANEL_VIEWPORT_RATIO),
        ),
      );
    }
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  return maxWidth;
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
    <>
      <div className="flex h-[53px] shrink-0 items-center gap-2 border-b border-b-[#80808017] px-3">
        <Avatar className="h-7 w-7 shrink-0">
          {expert.avatar_url ? (
            <AvatarImage
              src={expert.avatar_url}
              alt={expert.name}
              width={56}
              height={56}
            />
          ) : null}
          <AvatarFallback>{expert.name}</AvatarFallback>
        </Avatar>
        <h2 className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-900">
          {expert.name}&apos;s Soul
        </h2>
        <button
          type="button"
          aria-label="Close Soul panel"
          onClick={onClose}
          className="rounded p-1.5 text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
        >
          <Icon icon={Cancel01Icon} size={16} />
        </button>
      </div>

      <form onSubmit={save} className="flex min-h-0 flex-1 flex-col">
        <div className="relative min-h-0 flex-1">
          <div
            ref={setScrollElement}
            className="h-full overflow-y-auto px-5 py-5"
          >
            <Text variant="small" className="mb-5 text-zinc-500">
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
        <div className="flex shrink-0 justify-end gap-2 border-t border-t-[#80808017] px-5 py-3">
          <Button type="button" variant="ghost" size="small" onClick={onClose}>
            Cancel
          </Button>
          <Button
            type="submit"
            variant="primary"
            size="small"
            loading={isPending}
            disabled={!canSave}
          >
            Save Soul
          </Button>
        </div>
      </form>
    </>
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
        <Icon icon={LockIcon} size={16} className="text-zinc-500" />
        <SoulSectionTitle>Protected rules</SoulSectionTitle>
      </div>
      <div className="space-y-2 rounded-xl bg-zinc-50 p-4">
        {rules.map((rule) => (
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
        These rules are part of every expert&apos;s soul and cannot be edited.
      </Text>
    </section>
  );
}

function SoulSectionTitle({ children }: { children: ReactNode }) {
  return <h3 className="text-sm font-medium text-zinc-900">{children}</h3>;
}
