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

// Some resulable components

export const banner = (icon: string, content: string, className?: string) => `
  <div class="bg-violet-100 ring-1 ring-violet-500 rounded-2xl p-2 px-4 mt-2 flex items-start gap-2 text-sm font-medium text-purple-800 ${className || ""}">  
    <span class="flex-shrink-0">${icon}</span>
    <span>${content}</span>
  </div>
`;
