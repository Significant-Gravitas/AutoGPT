import type { Config } from "tailwindcss";
import plugin from "tailwindcss/plugin";
import tailwindcssAnimate from "tailwindcss-animate";
import { colors } from "./src/components/styles/colors";
import scrollbar from "./tailwind.scrollbar";
import tailwind3Preset from "./tailwind.v3-preset";

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

// Fractal-noise tile, inlined so the grain costs no extra request. `#` and `%`
// stay percent-encoded or the data URI terminates early.
const GRAIN_TEXTURE = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='grain'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23grain)'/%3E%3C/svg%3E")`;

// Lays grain over an element's own background. Deliberately sets no `position`,
// so it never fights the positioning utilities already on the target — callers
// must position the element themselves.
const grainTexture = plugin(function grainTexture({ addUtilities }) {
  addUtilities({
    ".grain-overlay::after": {
      content: '""',
      position: "absolute",
      inset: "0",
      backgroundImage: GRAIN_TEXTURE,
      opacity: "0.5",
      mixBlendMode: "soft-light",
      pointerEvents: "none",
    },
  });
});

const textShadow = plugin(function textShadow({ matchUtilities, theme }) {
  matchUtilities(
    {
      "text-shadow": (value) => ({ textShadow: value }),
    },
    { values: theme("textShadow") },
  );
});

// Every colour role is an RGB triplet in src/app/globals.css. Reading it
// through `rgb(... / <alpha-value>)` keeps opacity modifiers (`bg-surface/50`)
// working in Tailwind 3.
function role(name: string) {
  return `rgb(var(--color-${name}) / <alpha-value>)`;
}

const COLOR_ROLES = [
  "canvas",
  "surface",
  "surface-muted",
  "surface-hover",
  "surface-active",
  "overlay",
  "foreground-base",
  "foreground-strong",
  "foreground-emphasis",
  "foreground-body",
  "foreground-secondary",
  "foreground-tertiary",
  "foreground-subtle",
  "foreground-disabled",
  "placeholder",
  "separator",
  "border-base",
  "border-strong",
  "focus",
  "action",
  "action-hover",
  "action-foreground",
  "brand",
  "brand-hover",
  "accent-surface",
  "accent-strong",
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
  "builder-canvas",
  "builder-grid",
  "node-surface",
  "node-border",
  "node-selected",
  "edge",
  "handle",
] as const;

const roleColors = Object.fromEntries(
  COLOR_ROLES.map((name) => [name, role(name)]),
);

// Text tokens: one size per role, each bundled with its line-height, so a
// role is one decision instead of three. Values are what renders today: the
// Text atom's variants and the raw sizes in wide use. A raw size that sets no
// line-height today is swapped for `text-<role> leading-inherit`, so the
// bundled line-height never changes what renders.
//
// TODO(tokens): SECRT-3152. Left raw on purpose, still lint warnings: the
// rare sizes 5, 6, 9, 17, 23, 26, 35, 36, 48 and 64px, fractional rems
// (0.7, 0.8, 0.85em, 1.3rem, 12.5px), the Text `label` size 0.6785rem
// (10.856px) and its tracking, and the rem-based Text heading tracking
// (-0.033rem, -0.02rem, -0.01313rem, 0.06785rem).
const fontSize = {
  tiny: ["0.625rem", { lineHeight: "0.875rem" }],
  "caption-sm": ["0.6875rem", { lineHeight: "1rem" }],
  caption: ["0.75rem", { lineHeight: "1.125rem" }],
  "body-sm": ["0.8125rem", { lineHeight: "1.25rem" }],
  body: ["0.875rem", { lineHeight: "1.375rem" }],
  "body-md": ["0.9375rem", { lineHeight: "1.5rem" }],
  "body-lg": ["1rem", { lineHeight: "1.625rem" }],
  lead: ["1.25rem", { lineHeight: "1.75rem" }],
  "heading-sm": ["1.375rem", { lineHeight: "1.5rem" }],
  heading: ["1.75rem", { lineHeight: "2.5rem" }],
  title: ["2rem", { lineHeight: "2.5rem" }],
  display: ["2.75rem", { lineHeight: "3.5rem" }],
} satisfies Record<string, [string, { lineHeight: string }]>;

// Explicit overrides only. `inherit` reproduces a size class that never set a
// line-height; the 0.25rem steps cover raw values Tailwind's scale lacks.
const lineHeight = {
  inherit: "inherit",
  "4.5": "1.125rem",
  "5.5": "1.375rem",
  "6.5": "1.625rem",
};

const letterSpacing = {
  display: "-0.04em",
  heading: "-0.02em",
  title: "-0.01em",
  caps: "0.12em",
  eyebrow: "0.06em",
};

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
  darkMode: ["class", ".dark-mode"], // ignore dark: prefix classes for now until we fully support dark mode
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
      fontSize,
      lineHeight,
      letterSpacing,
      colors: {
        // The custom palette stays registered until DS5 has migrated every
        // directory: removing it would repaint each raw `zinc-*` class with
        // Tailwind's default palette.
        ...colors,
        ...roleColors,
        border: role("border"),
        input: role("input"),
        ring: role("ring"),
        background: role("background"),
        foreground: role("foreground"),
        primary: {
          DEFAULT: role("primary"),
          foreground: role("primary-foreground"),
        },
        secondary: {
          DEFAULT: role("secondary"),
          foreground: role("secondary-foreground"),
        },
        destructive: {
          DEFAULT: role("destructive"),
          foreground: role("destructive-foreground"),
        },
        muted: {
          DEFAULT: role("muted"),
          foreground: role("muted-foreground"),
        },
        accent: {
          DEFAULT: role("accent"),
          foreground: role("accent-foreground"),
        },
        popover: {
          DEFAULT: role("popover"),
          foreground: role("popover-foreground"),
        },
        card: {
          DEFAULT: role("card"),
          foreground: role("card-foreground"),
        },
        customGray: {
          "100": "#d9d9d9",
          "200": "#a8a8a8",
          "300": "#878787",
          "400": "#646464",
          "500": "#474747",
          "600": "#282828",
          "700": "#272727",
        },
        sidebar: {
          DEFAULT: role("sidebar"),
          foreground: role("sidebar-foreground"),
          primary: role("sidebar-primary"),
          "primary-foreground": role("sidebar-primary-foreground"),
          accent: role("sidebar-accent"),
          "accent-foreground": role("sidebar-accent-foreground"),
          border: role("sidebar-border"),
          ring: role("sidebar-ring"),
        },
      },
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
        "message-table": "0 1px 2px rgba(0,0,0,0.04)",
      },
      backgroundImage: {
        "message-table-header":
          "radial-gradient(35% 160% at 8% 0%, rgba(139,92,246,0.05), transparent), radial-gradient(35% 160% at 50% 0%, rgba(99,102,241,0.04), transparent), radial-gradient(35% 160% at 92% 0%, rgba(59,130,246,0.05), transparent), linear-gradient(to bottom, theme(colors.white), theme(colors.zinc.50))",
      },
      textShadow: {
        "message-table-violet":
          "0 0 6px rgba(139,92,246,0.45), 0 0 18px rgba(139,92,246,0.55)",
        "message-table-indigo":
          "0 0 6px rgba(99,102,241,0.45), 0 0 18px rgba(99,102,241,0.55)",
        "message-table-blue":
          "0 0 6px rgba(59,130,246,0.45), 0 0 18px rgba(59,130,246,0.55)",
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
  plugins: [
    tailwindcssAnimate,
    scrollbar,
    smoothShadowRing,
    grainTexture,
    textShadow,
  ],
} satisfies Config;

export default config;
