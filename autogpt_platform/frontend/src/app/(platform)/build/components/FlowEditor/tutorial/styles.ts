/**
 * Tutorial Styles for New Builder
 *
 * CSS file contains:
 * - Dynamic classes: .new-builder-tutorial-disable, .new-builder-tutorial-highlight, .new-builder-tutorial-pulse
 * - Shepherd.js overrides
 *
 * Typography (body, small, action, info, tip, warning) uses Tailwind utilities directly in steps.ts
 */
import "./tutorial.css";

export const injectTutorialStyles = () => {
  if (typeof window !== "undefined") {
    document.documentElement.setAttribute("data-tutorial-styles", "loaded");
  }
};

export const removeTutorialStyles = () => {
  if (typeof window !== "undefined") {
    document.documentElement.removeAttribute("data-tutorial-styles");
  }
};

// Reusable banner components with consistent styling

type BannerVariant = "action" | "info" | "warning" | "success";

const bannerStyles: Record<
  BannerVariant,
  { bg: string; ring: string; text: string }
> = {
  action: {
    bg: "bg-violet-50",
    ring: "ring-violet-200",
    text: "text-violet-800",
  },
  info: {
    bg: "bg-blue-50",
    ring: "ring-blue-200",
    text: "text-blue-800",
  },
  warning: {
    bg: "bg-amber-50",
    ring: "ring-amber-200",
    text: "text-amber-800",
  },
  success: {
    bg: "bg-green-50",
    ring: "ring-green-200",
    text: "text-green-800",
  },
};

export const banner = (
  icon: string,
  content: string,
  variant: BannerVariant = "action",
  className?: string,
) => {
  const styles = bannerStyles[variant];
  return `
  <div class="${styles.bg} ring-1 ${styles.ring} rounded-2xl p-2 px-4 mt-2 flex items-start gap-2 text-sm font-medium ${styles.text} ${className || ""}">  
    <span class="flex-shrink-0">${icon}</span>
    <span>${content}</span>
  </div>
`;
};

// Requirement box components
export const requirementBox = (
  title: string,
  items: string,
  variant: "warning" | "success" = "warning",
) => {
  const isSuccess = variant === "success";
  return `
  <div id="requirements-box" class="mt-3 p-3 ${isSuccess ? "bg-green-50 ring-1 ring-green-200" : "bg-amber-50 ring-1 ring-amber-200"} rounded-2xl">
    <p class="text-sm font-medium ${isSuccess ? "text-green-600" : "text-amber-600"} m-0 mb-2">${title}</p>
    ${items}
  </div>
`;
};

export const requirementItem = (id: string, content: string) => `
  <li id="${id}" class="flex items-center gap-2 text-amber-600">
    <span class="req-icon">○</span> ${content}
  </li>
`;

// Connection status box
export const connectionStatusBox = (
  id: string,
  variant: "waiting" | "connected" = "waiting",
) => {
  const isConnected = variant === "connected";
  return `
  <div id="${id}" class="mt-3 p-2 ${isConnected ? "bg-green-50 ring-1 ring-green-200" : "bg-amber-50 ring-1 ring-amber-200"} rounded-2xl text-center text-sm ${isConnected ? "text-green-600" : "text-amber-600"}">
    ${isConnected ? "✅ Connection already exists!" : "Waiting for connection..."}
  </div>
`;
};
