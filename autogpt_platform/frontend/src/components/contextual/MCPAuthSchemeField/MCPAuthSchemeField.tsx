"use client";

import { useId } from "react";

import type { MCPAuthScheme } from "@/lib/mcp-auth";

interface Props {
  value: MCPAuthScheme;
  onChange: (scheme: MCPAuthScheme) => void;
  disabled?: boolean;
  /** Disambiguates the accessible name when several connectors render at once. */
  nameSuffix?: string;
  className?: string;
  labelClassName?: string;
  selectClassName?: string;
}

/** Bearer/Basic selector shared by every surface that takes a manual MCP credential. */
export function MCPAuthSchemeField({
  value,
  onChange,
  disabled,
  nameSuffix,
  className,
  labelClassName,
  selectClassName,
}: Props) {
  const id = useId();
  const accessibleName = nameSuffix
    ? `Authentication type for ${nameSuffix}`
    : undefined;

  return (
    <div className={className}>
      <label htmlFor={id} className={labelClassName}>
        Authentication type
      </label>
      <select
        id={id}
        aria-label={accessibleName}
        value={value}
        onChange={(e) => onChange(e.target.value as MCPAuthScheme)}
        disabled={disabled}
        className={selectClassName}
      >
        <option value="bearer">API token (Bearer)</option>
        <option value="basic">Basic authentication</option>
      </select>
    </div>
  );
}
