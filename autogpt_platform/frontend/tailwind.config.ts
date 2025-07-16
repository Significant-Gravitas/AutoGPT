import type { Config } from "tailwindcss";
import tailwindcssAnimate from "tailwindcss-animate";
import { colors } from "./src/components/styles/colors";

const config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx}"],
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
        // *** APPROVED DESIGN SYSTEM COLORS ***
        // These are the ONLY colors that should be used in our app
        ...colors,

        // Legacy colors - DO NOT USE THESE IN NEW CODE
        // These are kept only to prevent breaking existing styles
        // Use the approved design system colors above instead
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
        customGray: {
          100: "#d9d9d9",
          200: "#a8a8a8",
          300: "#878787",
          400: "#646464",
          500: "#474747",
          600: "#282828",
          700: "#272727",
        },
      },
      spacing: {
        // Tailwind spacing + custom sizes
        0: "0rem", // 0px
        0.5: "0.125rem", // 2px
        1: "0.25rem", // 4px
        1.5: "0.375rem", // 6px
        2: "0.5rem", // 8px
        2.5: "0.625rem", // 10px
        3: "0.75rem", // 12px
        3.5: "0.875rem", // 14px
        4: "1rem", // 16px
        5: "1.25rem", // 20px
        6: "1.5rem", // 24px
        7: "1.75rem", // 28px
        7.5: "1.875rem", // 30px
        8: "2rem", // 32px
        8.5: "2.125rem", // 34px
        9: "2.25rem", // 36px
        10: "2.5rem", // 40px
        11: "2.75rem", // 44px
        12: "3rem", // 48px
        14: "3.5rem", // 56px
        16: "4rem", // 64px
        18: "4.5rem", // 72px
        20: "5rem", // 80px
        24: "6rem", // 96px
        28: "7rem", // 112px
        32: "8rem", // 128px
        36: "9rem", // 144px
        40: "10rem", // 160px
        44: "11rem", // 176px
        48: "12rem", // 192px
        52: "13rem", // 208px
        56: "14rem", // 224px
        60: "15rem", // 240px
        64: "16rem", // 256px
        68: "17rem", // 272px
        70: "17.5rem", // 280px
        71: "17.75rem", // 284px
        72: "18rem", // 288px
        76: "19rem", // 304px
        80: "20rem", // 320px
        96: "24rem", // 384px
      },
      borderRadius: {
        // Design system border radius tokens from Figma
        xsmall: "0.25rem", // 4px
        small: "0.5rem", // 8px
        medium: "0.75rem", // 12px
        large: "1rem", // 16px
        xlarge: "1.25rem", // 20px
        "2xlarge": "1.5rem", // 24px
        full: "9999px", // For pill buttons

        // Legacy values - kept for backward compatibility
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      boxShadow: {
        subtle: "0px 1px 2px 0px rgba(0,0,0,0.05)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "fade-in": {
          "0%": { opacity: "0" }, // Start with opacity 0
          "100%": { opacity: "1" }, // End with opacity 1
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "fade-in": "fade-in 0.2s ease-out",
      },
      transitionDuration: {
        "2000": "2000ms",
      },
    },
  },
  plugins: [tailwindcssAnimate],
} satisfies Config;

export default config;
