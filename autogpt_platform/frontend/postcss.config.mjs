import { createRequire } from "node:module";

// An absolute path: Next.js resolves plugin names from the project folder, but
// Storybook's postcss-loader does not.
const tailwind3Compat = createRequire(import.meta.url).resolve(
  "./postcss.tailwind3-compat.cjs",
);

/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    // Unminified, so colours keep the exact values written below instead of
    // being rounded into 8-bit hex.
    "@tailwindcss/postcss": { optimize: false },
    "postcss-nested": {},
    // Tailwind 3's output for the utilities Tailwind 4 changed; see the file.
    [tailwind3Compat]: {},
  },
};

export default config;
