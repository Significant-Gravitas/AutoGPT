/*
  The scrollbar utilities of tailwind-scrollbar 3.1.0 (MIT License,
  https://github.com/adoxography/tailwind-scrollbar), with `nocompatible`
  on, ported to Tailwind 4's plugin API. tailwind-scrollbar 3 reads
  Tailwind 3 internals and does not load under Tailwind 4, and version 4
  writes different CSS (no `!important` on colours, no hover and active
  thumb rules, colour roles left with an unresolved `<alpha-value>`).

  Its `hover:` / `focus:` / `active:` handling for `scrollbar-*` colours
  (set `--scrollbar-thumb-hover` rather than wait for `:hover`) needs to
  rewrite selectors, so it lives in postcss.tailwind3-compat.cjs.
*/
import plugin from "tailwindcss/plugin";

const COMPONENTS = ["track", "thumb", "corner"] as const;

type Palette = { [name: string]: string | Palette };

// Tailwind 3's flattenColorPalette: nested names joined with `-`, and a
// `DEFAULT` entry named after its group.
function flattenColors(colors: Palette, prefix = ""): Record<string, string> {
  return Object.fromEntries(
    Object.entries(colors).flatMap(([name, value]) => {
      const key =
        name === "DEFAULT" ? prefix : prefix ? `${prefix}-${name}` : name;
      return typeof value === "string"
        ? [[key, value]]
        : Object.entries(flattenColors(value, key));
    }),
  );
}

function webkitParts() {
  return Object.fromEntries(
    COMPONENTS.flatMap((component) => {
      const base = `&::-webkit-scrollbar-${component}`;
      return [
        [
          base,
          {
            "background-color": `var(--scrollbar-${component})`,
            "border-radius": `var(--scrollbar-${component}-radius)`,
          },
        ],
        [
          `${base}:hover`,
          {
            "background-color": `var(--scrollbar-${component}-hover, var(--scrollbar-${component}))`,
          },
        ],
        [
          `${base}:active`,
          {
            "background-color": `var(--scrollbar-${component}-active, var(--scrollbar-${component}-hover, var(--scrollbar-${component})))`,
          },
        ],
      ];
    }),
  );
}

const scrollbar = plugin(function scrollbar({
  addBase,
  addUtilities,
  matchUtilities,
  theme,
}) {
  addBase({
    "*": { "scrollbar-color": "initial", "scrollbar-width": "initial" },
  });

  addUtilities({
    ".scrollbar": {
      ...webkitParts(),
      "scrollbar-width": "auto",
      "scrollbar-color":
        "var(--scrollbar-thumb, initial) var(--scrollbar-track, initial)",
      "&::-webkit-scrollbar": {
        display: "block",
        width: "var(--scrollbar-width, 16px)",
        height: "var(--scrollbar-height, 16px)",
      },
    },
    ".scrollbar-thin": {
      ...webkitParts(),
      "scrollbar-width": "thin",
      "scrollbar-color":
        "var(--scrollbar-thumb, initial) var(--scrollbar-track, initial)",
      "&::-webkit-scrollbar": { display: "block", width: "8px", height: "8px" },
    },
    ".scrollbar-none": {
      "scrollbar-width": "none",
      "&::-webkit-scrollbar": { display: "none" },
    },
  });

  // Tailwind 3 filled a colour role's `<alpha-value>` with 1.
  const colors = Object.fromEntries(
    Object.entries(flattenColors(theme("colors") as Palette)).map(
      ([name, value]) => [name, value.replace("<alpha-value>", "1")],
    ),
  );
  for (const component of COMPONENTS) {
    matchUtilities(
      {
        [`scrollbar-${component}`]: (value: string) => ({
          [`--scrollbar-${component}`]: `${value.replace("<alpha-value>", "1")} !important`,
        }),
      },
      { values: colors, type: "color" },
    );
    matchUtilities(
      {
        [`scrollbar-${component}-rounded`]: (value: string) => ({
          [`--scrollbar-${component}-radius`]: value,
        }),
      },
      { values: theme("borderRadius") },
    );
  }
  for (const dimension of ["width", "height"] as const) {
    matchUtilities(
      {
        [`scrollbar-${dimension[0]}`]: (value: string) => ({
          [`--scrollbar-${dimension}`]: value,
        }),
      },
      { values: theme(dimension) },
    );
  }
});

export default scrollbar;
