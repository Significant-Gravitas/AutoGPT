"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import { AUTOPILOT_AVATAR } from "@/components/molecules/BotAvatar/helpers";

import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { SelectableCard } from "../components/SelectableCard";
import { useOnboardingWizardStore } from "../store";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  ChartLineData01Icon,
  CodeIcon,
  CubeIcon,
  Flag01Icon,
  Megaphone01Icon,
  Settings02Icon,
  Target01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";

const ICON_SIZE = 20;

const ROLES = [
  {
    id: "Founder/CEO",
    label: "Founder / CEO",
    icon: <Icon icon={Target01Icon} size={ICON_SIZE} />,
  },
  {
    id: "Operations",
    label: "Operations",
    icon: <Icon icon={Settings02Icon} size={ICON_SIZE} />,
  },
  {
    id: "Sales/BD",
    label: "Sales / BD",
    icon: <Icon icon={ChartLineData01Icon} size={ICON_SIZE} />,
  },
  {
    id: "Marketing",
    label: "Marketing",
    icon: <Icon icon={Megaphone01Icon} size={ICON_SIZE} />,
  },
  {
    id: "Product/PM",
    label: "Product / PM",
    icon: <Icon icon={CubeIcon} size={ICON_SIZE} />,
  },
  {
    id: "Engineering",
    label: "Engineering",
    icon: <Icon icon={CodeIcon} size={ICON_SIZE} />,
  },
  {
    id: "HR/People",
    label: "HR / People",
    icon: <Icon icon={UserGroupIcon} size={ICON_SIZE} />,
  },
  {
    id: "Other",
    label: "Other",
    icon: <Icon icon={Flag01Icon} size={ICON_SIZE} />,
  },
] as const;

export function RoleStep() {
  const role = useOnboardingWizardStore((s) => s.role);
  const otherRole = useOnboardingWizardStore((s) => s.otherRole);
  const setRole = useOnboardingWizardStore((s) => s.setRole);
  const setOtherRole = useOnboardingWizardStore((s) => s.setOtherRole);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const reduceMotion = useReducedMotion();

  const isOther = role === "Other";
  const canContinue = isOther ? Boolean(otherRole.trim()) : Boolean(role);

  function handleNext() {
    if (canContinue) nextStep();
  }

  return (
    <FadeIn>
      <div className="flex w-full flex-col items-center gap-8 px-4">
        <div className="mx-auto flex w-full max-w-lg flex-col items-center gap-4 px-4 text-center">
          <BotAvatar
            config={AUTOPILOT_AVATAR}
            status="idle"
            size={120}
            trackPointer
            showBadge={false}
          />
          <Text variant="h4">What best describes you?</Text>
        </div>

        <div className="flex w-full max-w-[100vw] flex-nowrap gap-4 overflow-x-auto px-8 scrollbar-none md:grid md:grid-cols-4 md:overflow-hidden md:px-0">
          {ROLES.map((r) => (
            <SelectableCard
              key={r.id}
              icon={r.icon}
              label={r.label}
              selected={role === r.id}
              onClick={() => setRole(r.id)}
              className="h-28 w-[11.5rem]"
            />
          ))}
        </div>

        <AnimatePresence initial={false}>
          {isOther && (
            <motion.div
              key="other-role"
              initial={{ opacity: 0, height: 0, y: reduceMotion ? 0 : -8 }}
              animate={{ opacity: 1, height: "auto", y: 0 }}
              exit={{ opacity: 0, height: 0, y: reduceMotion ? 0 : -8 }}
              transition={{
                duration: reduceMotion ? 0.15 : 0.24,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="-my-1 w-full max-w-lg overflow-hidden"
            >
              <div className="w-full px-8 py-1 md:px-0">
                <Input
                  id="other-role"
                  label="Other role"
                  hideLabel
                  placeholder="Describe your role..."
                  value={otherRole}
                  onChange={(e) => setOtherRole(e.target.value)}
                  autoFocus
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <Button
          type="button"
          size="small"
          onClick={handleNext}
          disabled={!canContinue}
          className="h-10 w-56 rounded-xl"
        >
          Next
        </Button>
      </div>
    </FadeIn>
  );
}
