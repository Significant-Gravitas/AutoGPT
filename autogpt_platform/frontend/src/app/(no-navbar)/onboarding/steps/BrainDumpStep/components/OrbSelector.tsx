"use client";

import { Select } from "@/components/atoms/Select/Select";

export type OrbVariant = "glass" | "wavy" | "orb-ui";

const ORB_OPTIONS = [
  { value: "glass", label: "Current orb" },
  { value: "wavy", label: "Wavy orb" },
  { value: "orb-ui", label: "Orb UI" },
];

interface Props {
  value: OrbVariant;
  onChange: (value: OrbVariant) => void;
}

export function OrbSelector({ value, onChange }: Props) {
  function handleChange(nextValue: string) {
    if (
      nextValue === "glass" ||
      nextValue === "wavy" ||
      nextValue === "orb-ui"
    ) {
      onChange(nextValue);
    }
  }

  return (
    <Select
      id="onboarding-orb-style"
      label="Orb style"
      hideLabel
      size="small"
      value={value}
      onValueChange={handleChange}
      options={ORB_OPTIONS}
      className="w-32 sm:w-40"
      wrapperClassName="!mb-0"
    />
  );
}
