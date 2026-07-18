"use client";

import { Select } from "@/components/atoms/Select/Select";
import { InformationTooltip } from "@/components/molecules/InformationTooltip/InformationTooltip";
import { cn } from "@/lib/utils";

import {
  ORG_ROLE_OPTIONS,
  roleCapabilitiesMarkdown,
  type OrgRole,
} from "./roleAccess";

interface Props {
  id: string;
  ariaLabel: string;
  value: OrgRole;
  onChange: (role: OrgRole) => void;
  disabled?: boolean;
  size?: "small" | "medium";
  selectClassName?: string;
}

export function OrgRoleSelect({
  id,
  ariaLabel,
  value,
  onChange,
  disabled,
  size = "small",
  selectClassName,
}: Props) {
  return (
    <div className="flex items-center gap-1">
      <Select
        id={id}
        label={ariaLabel}
        hideLabel
        size={size}
        wrapperClassName={cn("!mb-0 w-44", selectClassName)}
        value={value}
        onValueChange={(role) => onChange(role as OrgRole)}
        options={ORG_ROLE_OPTIONS}
        disabled={disabled}
      />
      <InformationTooltip
        description={roleCapabilitiesMarkdown(value)}
        iconSize={20}
      />
    </div>
  );
}
