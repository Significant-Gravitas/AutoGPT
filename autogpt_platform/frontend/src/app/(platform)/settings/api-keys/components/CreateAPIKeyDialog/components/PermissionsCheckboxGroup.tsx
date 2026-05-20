"use client";

import { CheckSquareIcon, SquareIcon } from "@phosphor-icons/react";

import type { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import { Text } from "@/components/atoms/Text/Text";

import { PERMISSION_OPTIONS } from "../schema";

interface Props {
  value: APIKeyPermission[];
  onChange: (next: APIKeyPermission[]) => void;
}

export function PermissionsCheckboxGroup({ value, onChange }: Props) {
  function toggle(permission: APIKeyPermission) {
    if (value.includes(permission)) {
      onChange(value.filter((p) => p !== permission));
    } else {
      onChange([...value, permission]);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <Text
        id="api-key-permissions-label"
        variant="large-medium"
        as="span"
        className="text-textBlack"
      >
        Permissions
      </Text>
      <div
        role="group"
        aria-labelledby="api-key-permissions-label"
        className="grid max-h-[220px] grid-cols-2 gap-x-4 gap-y-2 overflow-y-auto"
      >
        {PERMISSION_OPTIONS.map((option) => {
          const checked = value.includes(option.value);
          return (
            <button
              key={option.value}
              type="button"
              role="checkbox"
              aria-checked={checked}
              onClick={() => toggle(option.value)}
              className="flex items-center gap-2 rounded text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
            >
              {checked ? (
                <CheckSquareIcon size={18} weight="fill" />
              ) : (
                <SquareIcon size={18} />
              )}
              <Text variant="body" as="span" className="text-zinc-700">
                {option.label}
              </Text>
            </button>
          );
        })}
      </div>
    </div>
  );
}
