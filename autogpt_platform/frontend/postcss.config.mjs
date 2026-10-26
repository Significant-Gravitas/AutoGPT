import { fileURLToPath } from "node:url";

// An absolute path: Next.js resolves plugin names from the project folder, but
// Storybook's postcss-loader does not.
const tailwind3Compat = fileURLToPath(
  new URL("./postcss.tailwind3-compat.cjs", import.meta.url),
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
