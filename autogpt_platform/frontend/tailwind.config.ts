import type { Config } from "tailwindcss";
import tailwindcssAnimate from "tailwindcss-animate";
import tailwind3Preset from "./tailwind.v3-preset.ts";

// The colours, type scale and radii are the theme in src/app/globals.css,
// with this app's own utilities (`@utility`). What is left here is the rest of
// Tailwind 3's theme (the preset), the spacing, shadow and motion values
// still in JS, and tailwindcss-animate.

// Classes Tailwind 3 generated nothing for, which Tailwind 4 would now turn
// into CSS: bare numbers it accepts without a theme step (`z-12`, `flex-2`,
// `opacity-1`), an `ease-[...]` arbitrary value Tailwind 3 rejected as
// ambiguous, and streamdown's Tailwind 4 syntax (`content-(--x)`, a trailing
// `!`, `wrap-anywhere`). Listing them keeps every element looking as it did;
// the owning components can drop or fix them on their own.
const TAILWIND_3_DEAD_CLASSES = [
  "z-12",
  "flex-2",
  "disabled:opacity-1",
  "ease-[cubic-bezier(0.16,1,0.3,1)]",
  "ease-[cubic-bezier(0.33,1,0.68,1)]",
  "data-[state=open]:ease-[cubic-bezier(0.16,1,0.3,1)]",
  "data-[state=closed]:ease-[cubic-bezier(0.4,0,1,1)]",
  "wrap-anywhere",
  "*:last:after:content-(--streamdown-caret)",
  // Streamdown's classes below stack variants in Tailwind 4's order, which
  // reads them the other way round from Tailwind 3; globals.css keeps the
  // rules Tailwind 3 generated for them.
  "*:first:mt-0",
  "*:last:mb-0",
  "*:last:after:inline",
  "*:last:after:align-baseline",
  "dark:bg-(--shiki-dark-bg)!",
  "dark:text-(--shiki-dark)!",
];

const config = {
  presets: [tailwind3Preset],
  blocklist: TAILWIND_3_DEAD_CLASSES,
  content: ["./src/**/*.{ts,tsx}", "./node_modules/streamdown/dist/**/*.js"],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      spacing: {
        "0": "0rem",
        "1": "0.25rem",
        "2": "0.5rem",
        "3": "0.75rem",
        "4": "1rem",
        "5": "1.25rem",
        "6": "1.5rem",
        "7": "1.75rem",
        "8": "2rem",
        "9": "2.25rem",
        "10": "2.5rem",
        "11": "2.75rem",
        "12": "3rem",
        "14": "3.5rem",
        "16": "4rem",
        "18": "4.5rem",
        "20": "5rem",
        "24": "6rem",
        "28": "7rem",
        "32": "8rem",
        "36": "9rem",
        "40": "10rem",
        "44": "11rem",
        "48": "12rem",
        "52": "13rem",
        "56": "14rem",
        "60": "15rem",
        "64": "16rem",
        "68": "17rem",
        "70": "17.5rem",
        "71": "17.75rem",
        "72": "18rem",
        "76": "19rem",
        "80": "20rem",
        "96": "24rem",
        "0.5": "0.125rem",
        "1.5": "0.375rem",
        "2.5": "0.625rem",
        "3.5": "0.875rem",
        "4.5": "1.125rem",
        "7.5": "1.875rem",
        "8.5": "2.125rem",
      },
      boxShadow: {
        subtle: "0px 1px 2px 0px rgba(0,0,0,0.05)",
        "message-table": "0 1px 2px rgba(0,0,0,0.04)",
      },
      backgroundImage: {
        "message-table-header":
          "radial-gradient(35% 160% at 8% 0%, rgba(139,92,246,0.05), transparent), radial-gradient(35% 160% at 50% 0%, rgba(99,102,241,0.04), transparent), radial-gradient(35% 160% at 92% 0%, rgba(59,130,246,0.05), transparent), linear-gradient(to bottom, theme(colors.white), theme(colors.zinc.50))",
      },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
        // Bridge the height change with an opacity fade so the content
        // doesn't clip in/out abruptly (Emil Kowalski: fade to soften
        // state transitions). Opacity resolves over the first/last ~60%
        // so text is fully legible before the height finishes settling.
        "collapsible-down": {
          from: {
            height: "0",
            opacity: "0",
          },
          "60%": {
            opacity: "1",
          },
          to: {
            height: "var(--radix-collapsible-content-height)",
            opacity: "1",
          },
        },
        "collapsible-up": {
          from: {
            height: "var(--radix-collapsible-content-height)",
            opacity: "1",
          },
          "40%": {
            opacity: "0",
          },
          to: {
            height: "0",
            opacity: "0",
          },
        },
        "fade-in": {
          "0%": {
            opacity: "0",
          },
          "100%": {
            opacity: "1",
          },
        },
        shimmer: {
          "0%": {
            backgroundPosition: "200% 0",
          },
          "100%": {
            backgroundPosition: "-200% 0",
          },
        },
        loader: {
          "0%": {
            boxShadow: "0 0 0 0 rgba(0, 0, 0, 0.25)",
          },
          "100%": {
            boxShadow: "0 0 0 30px rgba(0, 0, 0, 0)",
          },
        },
        shake: {
          "0%, 100%": { transform: "translateX(0)" },
          "20%": { transform: "translateX(-4px)" },
          "40%": { transform: "translateX(4px)" },
          "60%": { transform: "translateX(-3px)" },
          "80%": { transform: "translateX(3px)" },
        },
        "marquee-x": {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        "progress-bar": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(400%)" },
        },
        "caret-blink": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0" },
        },
        "shimmer-text": {
          "0%": { backgroundPosition: "100% 0" },
          "100%": { backgroundPosition: "0% 0" },
        },
        "fade-up": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "grow-line": {
          from: { transform: "scaleY(0)" },
          to: { transform: "scaleY(1)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        // easeOutCubic for a smooth, decelerating settle; asymmetric
        // timing (open a touch slower than close) per Emil Kowalski.
        // Both stay under the 300ms UI budget.
        "collapsible-down":
          "collapsible-down 0.26s cubic-bezier(0.33, 1, 0.68, 1)",
        "collapsible-up": "collapsible-up 0.2s cubic-bezier(0.33, 1, 0.68, 1)",
        "fade-in": "fade-in 0.2s ease-out",
        shimmer: "shimmer 4s ease-in-out infinite",
        loader: "loader 1s infinite",
        shake: "shake 0.5s ease-in-out",
        "marquee-x": "marquee-x 40s linear infinite",
        "progress-bar":
          "progress-bar 1.4s cubic-bezier(0.65, 0, 0.35, 1) infinite",
        "caret-blink": "caret-blink 1s step-end infinite",
        "shimmer-text": "shimmer-text 2s linear infinite",
        "fade-up": "fade-up 320ms cubic-bezier(0.23, 1, 0.32, 1) both",
        "grow-line": "grow-line 500ms cubic-bezier(0.23, 1, 0.32, 1) both",
      },
      transitionDuration: {
        "400": "400ms",
        "2000": "2000ms",
      },
      transitionTimingFunction: {
        // easeOutQuint — long, soft settle for accordion expand/collapse.
        "out-quint": "cubic-bezier(0.23, 1, 0.32, 1)",
      },
    },
  },
  plugins: [tailwindcssAnimate],
} satisfies Config;

export default config;
