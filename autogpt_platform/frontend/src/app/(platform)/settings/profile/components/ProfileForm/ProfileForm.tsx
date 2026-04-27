"use client";

import { motion, useReducedMotion } from "framer-motion";

import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";

import {
  MAX_BIO_LENGTH,
  type ProfileFormState,
  validateForm,
} from "../../helpers";

interface Props {
  formState: ProfileFormState;
  errors: ReturnType<typeof validateForm>["errors"];
  onChange: <K extends keyof ProfileFormState>(
    key: K,
    value: ProfileFormState[K],
  ) => void;
}

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

export function ProfileForm({ formState, errors, onChange }: Props) {
  const reduceMotion = useReducedMotion();
  const remaining = MAX_BIO_LENGTH - formState.description.length;
  const counterColor =
    remaining < 0
      ? "text-red-500"
      : remaining < 30
        ? "text-amber-600"
        : "text-zinc-400";

  return (
    <motion.div
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT, delay: 0.06 }}
      className="flex flex-col gap-5 rounded-[16px] border border-zinc-200 bg-white p-6 shadow-[0_1px_2px_rgba(15,15,20,0.04)]"
    >
      <div className="flex flex-col gap-1">
        <Text variant="h4" as="h2" className="text-[#1F1F20]">
          About you
        </Text>
        <Text variant="small" className="text-zinc-500">
          Public details shown on your marketplace profile.
        </Text>
      </div>

      <Input
        id="profile-name"
        label="Display name"
        placeholder="Jane Doe"
        value={formState.name}
        error={errors.name}
        onChange={(e) => onChange("name", e.target.value)}
      />

      <Input
        id="profile-username"
        label="Handle"
        placeholder="jane_doe"
        hint="autogpt.com/@your-handle"
        value={formState.username}
        error={errors.username}
        onChange={(e) => onChange("username", e.target.value)}
      />

      <div className="flex w-full flex-col gap-2">
        <div className="flex items-center justify-between">
          <Text variant="large-medium" as="span" className="text-black">
            Bio
          </Text>
          <Text
            variant="small"
            as="span"
            className={`tabular-nums transition-colors duration-150 ${counterColor}`}
          >
            {Math.max(remaining, 0)} left
          </Text>
        </div>
        <Input
          id="profile-bio"
          label="Bio"
          hideLabel
          type="textarea"
          rows={5}
          placeholder="Tell people what you build, the agents you ship, and what you care about."
          value={formState.description}
          error={errors.description}
          onChange={(e) => onChange("description", e.target.value)}
        />
      </div>
    </motion.div>
  );
}
