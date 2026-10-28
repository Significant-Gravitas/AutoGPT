"use client";
import type { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import { Text } from "@/components/atoms/Text/Text";

import { PERMISSION_GROUPS } from "../schema";
import { CheckmarkSquare02Icon, SquareIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
        className="flex max-h-[min(320px,30vh)] flex-col gap-5 overflow-y-auto overscroll-contain pr-1"
      >
        {PERMISSION_GROUPS.map((group) => (
          <PermissionGroup
            key={group.version}
            group={group}
            value={value}
            onToggle={toggle}
          />
        ))}
      </div>
    </div>
  );
}

interface GroupProps {
  group: (typeof PERMISSION_GROUPS)[number];
  value: APIKeyPermission[];
  onToggle: (permission: APIKeyPermission) => void;
}

function PermissionGroup({ group, value, onToggle }: GroupProps) {
  const headingID = `api-key-scopes-${group.version}`;

  return (
    <div
      role="group"
      aria-labelledby={headingID}
      className="flex flex-col gap-1"
    >
      <Text
        id={headingID}
        variant="small-medium"
        as="span"
        className="text-foreground-secondary"
      >
        {group.label}
      </Text>
      {group.options.map((option) => {
        const checked = value.includes(option.value);
        const labelID = `api-key-scope-${option.value}`;
        return (
          <button
            key={option.value}
            type="button"
            role="checkbox"
            aria-checked={checked}
            aria-labelledby={labelID}
            aria-describedby={`${labelID}-description`}
            onClick={() => onToggle(option.value)}
            className="-mx-1 flex items-start gap-2 rounded-md px-1 py-1 text-left hover:bg-surface-hover focus:outline-none focus-visible:ring-2 focus-visible:ring-action"
          >
            <Icon
              icon={checked ? CheckmarkSquare02Icon : SquareIcon}
              size={18}
              className="mt-0.5 shrink-0"
              aria-hidden="true"
            />
            <span className="flex min-w-0 flex-col">
              <Text
                id={labelID}
                variant="body"
                as="span"
                className="text-foreground-body"
              >
                {option.label}
              </Text>
              <Text
                id={`${labelID}-description`}
                variant="small"
                as="span"
                className="text-foreground-secondary"
              >
                {option.description}
              </Text>
            </span>
          </button>
        );
      })}
    </div>
  );
}
