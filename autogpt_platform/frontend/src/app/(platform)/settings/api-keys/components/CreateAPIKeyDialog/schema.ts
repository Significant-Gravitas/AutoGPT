import { z } from "zod";

import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import {
  SCOPE_API_VERSION,
  SCOPE_DESCRIPTIONS,
  SCOPE_GROUPS,
} from "@/lib/api-key-scopes";

export const createAPIKeySchema = z.object({
  name: z
    .string()
    .trim()
    .min(1, "Name is required")
    .max(100, "Name must be 100 characters or less"),
  description: z
    .string()
    .trim()
    .max(500, "Description must be 500 characters or less")
    .optional(),
  permissions: z
    .array(z.nativeEnum(APIKeyPermission))
    .min(1, "Select at least one permission"),
});

export type CreateAPIKeyFormValues = z.infer<typeof createAPIKeySchema>;

export function humanizePermission(permission: APIKeyPermission): string {
  return permission
    .toLowerCase()
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

export const PERMISSION_GROUPS = SCOPE_GROUPS.map((group) => ({
  ...group,
  options: Object.values(APIKeyPermission)
    .filter((permission) => SCOPE_API_VERSION[permission] === group.version)
    .map((permission) => ({
      value: permission,
      label: humanizePermission(permission),
      description: SCOPE_DESCRIPTIONS[permission],
    })),
}));
