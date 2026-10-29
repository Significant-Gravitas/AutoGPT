import {
  extendTailwindMerge,
  getDefaultConfig,
  validators,
} from "tailwind-merge";

const defaults = getDefaultConfig();

// A quoted arbitrary font (`font-['Geist']`) is a family, as tailwind-merge 2
// read it; 3 reads it as a weight unless it says `family-name:`.
function isQuotedFamily(value: string) {
  return /^\[['"]/.test(value);
}

const fontWeights = defaults.classGroups["font-weight"].map((group) =>
  typeof group === "object" && "font" in group
    ? {
        font: group.font.map((definition) =>
          typeof definition === "function" && !("isThemeGetter" in definition)
            ? (value: string) => !isQuotedFamily(value) && definition(value)
            : definition,
        ),
      }
    : group,
);

// Teach tailwind-merge the text token names in tailwind.config.ts, so that
// `text-body` is read as a font size rather than a colour (and survives next
// to `text-zinc-500`) and conflicting sizes, line-heights and letter-spacings
// collapse to the last one. The radius names do the same for every
// `rounded-*` corner group, so `rounded-xl` and `rounded-2xlarge` no longer
// both survive and leave the winner to stylesheet order.
//
// The overrides keep tailwind-merge 3 merging the way 2 did, since the
// classes still render as in Tailwind 3 (see postcss.tailwind3-compat.cjs):
// - a bare `outline` sets only the style, so it survives next to `outline-2`;
// - `bg-gradient-to-*` is an image, so it no longer replaces `bg-<colour>`;
// - `border-x` / `border-y` survive next to `border-2`, and `max-h-none` next
//   to another `max-h-*`, leaving the winner to stylesheet order as before;
// - the `scrollbar-*` classes are tailwind.scrollbar.ts's, not Tailwind 4's
//   scrollbar utilities, so none of them replaces another.
export const twMerge = extendTailwindMerge({
  override: {
    classGroups: {
      "outline-w": [
        {
          outline: [
            validators.isNumber,
            validators.isArbitraryVariableLength,
            validators.isArbitraryLength,
          ],
        },
      ],
      "font-weight": fontWeights,
      "max-h": [
        {
          "max-h": defaults.classGroups["max-h"].flatMap((group) =>
            typeof group === "object" && "max-h" in group
              ? group["max-h"].filter((value) => value !== "none")
              : [],
          ),
        },
      ],
      "scrollbar-thumb-color": [],
      "scrollbar-track-color": [],
      "scrollbar-w": [],
      "scrollbar-gutter": [],
    },
    conflictingClassGroups: {
      "border-w": defaults.conflictingClassGroups["border-w"].filter(
        (group) => group !== "border-w-x" && group !== "border-w-y",
      ),
    },
  },
  extend: {
    theme: {
      radius: ["xsmall", "small", "medium", "large", "xlarge", "2xlarge"],
    },
    classGroups: {
      "font-size": [
        {
          text: [
            "tiny",
            "caption-sm",
            "caption",
            "body-sm",
            "body",
            "body-md",
            "body-lg",
            "lead",
            "heading-sm",
            "heading",
            "title",
            "display",
          ],
        },
      ],
      "font-family": [{ font: [isQuotedFamily] }],
      leading: [{ leading: ["inherit"] }],
      "outline-style": [{ outline: [""] }],
      "bg-image": [
        {
          "bg-gradient-to": ["t", "tr", "r", "br", "b", "bl", "l", "tl"],
        },
      ],
      tracking: [
        { tracking: ["display", "heading", "title", "caps", "eyebrow"] },
      ],
    },
  },
});
