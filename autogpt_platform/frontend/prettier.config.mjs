/** @type {import("prettier").Config} */
const config = {
  plugins: ["prettier-plugin-tailwindcss"],
  tailwindStylesheet: "./src/app/globals.css",
  // Sort classes inside these calls too, not only in className attributes.
  tailwindFunctions: ["cn", "cva", "clsx"],
  overrides: [
    {
      // Files whose cva()/cn() strings are not sorted yet. Sorting them is a
      // pure reorder; it is left to whoever next edits each file, and the
      // file then comes off this list. Do not add to it.
      files: [
        "src/components/__legacy__/Button.tsx",
        "src/components/__legacy__/ui/badge.tsx",
        "src/components/__legacy__/ui/button.tsx",
        "src/components/__legacy__/ui/calendar.tsx",
        "src/components/atoms/Button/helpers.ts",
        "src/components/atoms/DateInput/DateInput.tsx",
        "src/components/atoms/DateTimeInput/DateTimeInput.tsx",
        "src/components/atoms/Input/Input.tsx",
        "src/components/atoms/Select/Select.tsx",
        "src/components/molecules/Alert/Alert.tsx",
        "src/components/ui/button.tsx",
        "src/components/ui/input-group.tsx",
        "src/components/ui/sidebar.tsx",
      ],
      options: { tailwindFunctions: [] },
    },
  ],
};

export default config;
