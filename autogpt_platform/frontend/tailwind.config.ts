import scrollbar from "tailwind-scrollbar";
import type { Config } from "tailwindcss";
import plugin from "tailwindcss/plugin";
import tailwindcssAnimate from "tailwindcss-animate";
import { colors } from "./src/components/styles/colors";

const SMOOTH_SHADOW_SM =
  "0 18px 47px 0 color-mix(in srgb, var(--smooth-shadow-color) 3%, transparent), 0 7.5px 19px 0 color-mix(in srgb, var(--smooth-shadow-color) 2%, transparent), 0 4px 10.5px 0 color-mix(in srgb, var(--smooth-shadow-color) 2%, transparent), 0 2.3px 5.8px 0 color-mix(in srgb, var(--smooth-shadow-color) 1%, transparent), 0 1.2px 3.1px 0 color-mix(in srgb, var(--smooth-shadow-color) 1%, transparent), 0 0.5px 1.3px 0 color-mix(in srgb, var(--smooth-shadow-color) 1%, transparent)";
const SMOOTH_RING =
  "0 0 0 var(--smooth-ring-width, 1px) var(--smooth-ring-color)";

const smoothShadowRing = plugin(function smoothShadowRing({
  addBase,
  addUtilities,
}) {
  addBase({
    ":root": {
      "--smooth-ring-color": "rgba(0, 0, 0, 0.05)",
      "--smooth-ring-width": "1px",
    },
    '.dark, .dark-mode, [data-theme="dark"]': {
      "--smooth-ring-color": "rgba(255, 255, 255, 0.18)",
    },
  });
  addUtilities({
    ".smooth-shadow-ring-sm": {
      "--smooth-shadow-color": "var(--tw-shadow-color, rgb(0 0 0))",
      boxShadow: `${SMOOTH_SHADOW_SM}, ${SMOOTH_RING}`,
    },
  });
});

const config = {
  // Same selector as the .dark variables in globals.css and next-themes
  // attribute="class"; providers.tsx forces light, so dark: never applies yet.
  darkMode: "class",
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
      fontFamily: {
        sans: ["var(--font-geist-sans)"],
        mono: ["var(--font-geist-mono)"],
        poppins: ["var(--font-poppins)"],
      },
      colors: {
        ...colors,
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        sidebar: {
          DEFAULT: "hsl(var(--sidebar-background))",
          foreground: "hsl(var(--sidebar-foreground))",
          primary: "hsl(var(--sidebar-primary))",
          "primary-foreground": "hsl(var(--sidebar-primary-foreground))",
          accent: "hsl(var(--sidebar-accent))",
          "accent-foreground": "hsl(var(--sidebar-accent-foreground))",
          border: "hsl(var(--sidebar-border))",
          ring: "hsl(var(--sidebar-ring))",
        },
      },
      spacing: {
        "4.5": "1.125rem",
        "18": "4.5rem",
        "68": "17rem",
        "71": "17.75rem",
        "76": "19rem",
      },
      borderRadius: {
        xsmall: "0.25rem",
        small: "0.5rem",
        medium: "0.75rem",
        large: "1rem",
        xlarge: "1.25rem",
        "2xlarge": "1.5rem",
        full: "9999px",
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      boxShadow: {
        subtle: "0px 1px 2px 0px rgba(0,0,0,0.05)",
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
        shake: {
          "0%, 100%": { transform: "translateX(0)" },
          "20%": { transform: "translateX(-4px)" },
          "40%": { transform: "translateX(4px)" },
          "60%": { transform: "translateX(-3px)" },
          "80%": { transform: "translateX(3px)" },
        },
        aurora: {
          "0%": { backgroundPosition: "50% 50%, 50% 50%" },
          "100%": { backgroundPosition: "350% 50%, 350% 50%" },
        },
        "progress-bar": {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(400%)" },
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
        shake: "shake 0.5s ease-in-out",
        aurora: "aurora 60s linear infinite",
        "progress-bar":
          "progress-bar 1.4s cubic-bezier(0.65, 0, 0.35, 1) infinite",
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
  plugins: [
    tailwindcssAnimate,
    scrollbar({ nocompatible: true }),
    smoothShadowRing,
  ],
} satisfies Config;

export default config;
