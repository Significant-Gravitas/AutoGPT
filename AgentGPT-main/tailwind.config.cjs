/** @type {import("tailwindcss").Config} */
const defaultTheme = require("tailwindcss/defaultTheme");

module.exports = {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    screens: {
      "xs": "300px",

      "sm-h": { "raw": "(min-height: 700px)" },
      "md-h": { "raw": "(min-height: 800px)" },
      "lg-h": { "raw": "(min-height: 1000px)" },

      ...defaultTheme.screens
    },
    extend: {
      boxShadow: {
        "3xl": "0 40px 70px -15px rgba(0, 0, 0, 0.40)" // Customize the shadow value according to your preferences.
      }
    }
  },
  plugins: []
};
