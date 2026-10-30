// The colour roles, grouped by job. Each is an RGB triplet in
// src/app/globals.css (`--color-<name>`), which its `@theme` registers as a
// Tailwind colour; the Tokens/Colors story draws them from here.
export const COLOR_ROLE_GROUPS = [
  {
    name: "Surfaces",
    description: "Page and card backgrounds, hover and pressed fills.",
    roles: [
      "canvas",
      "surface",
      "surface-muted",
      "surface-hover",
      "surface-active",
      "overlay",
    ],
  },
  {
    name: "Text",
    description: "Text and icon colours, strongest first.",
    roles: [
      "foreground-base",
      "foreground-strong",
      "foreground-emphasis",
      "foreground-body",
      "foreground-secondary",
      "foreground-tertiary",
      "foreground-subtle",
      "foreground-disabled",
      "placeholder",
    ],
  },
  {
    name: "Borders",
    description: "Dividers, borders and the focus ring.",
    roles: ["separator", "border-base", "border-strong", "focus"],
  },
  {
    name: "Action",
    description: "The primary Button and the text on it.",
    roles: ["action", "action-hover", "action-foreground"],
  },
  {
    name: "Accent",
    description: "The brand purple.",
    roles: ["brand", "brand-hover", "accent-surface", "accent-strong"],
  },
  {
    name: "Status",
    description: "Error, warning, success and info.",
    roles: [
      "error",
      "error-hover",
      "error-foreground",
      "error-surface",
      "warning",
      "warning-foreground",
      "warning-surface",
      "warning-border",
      "success",
      "success-foreground",
      "success-surface",
      "info",
      "info-foreground",
      "info-surface",
    ],
  },
  {
    name: "Builder",
    description: "The flow editor's canvas, nodes, edges and handles.",
    roles: [
      "builder-canvas",
      "builder-grid",
      "node-surface",
      "node-border",
      "node-selected",
      "edge",
      "handle",
    ],
  },
] as const;

export const COLOR_ROLES = COLOR_ROLE_GROUPS.flatMap((group) => group.roles);
