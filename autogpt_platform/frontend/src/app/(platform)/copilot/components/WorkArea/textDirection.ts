/** Hebrew, Arabic, Syriac, Thaana and N'Ko letters, and their forms. */
const RTL_LETTER = /[֐-ࣿיִ-﷿ﹰ-﻿]/u;
const FIRST_LETTER = /\p{L}/u;

/** The text reads right to left: its first letter is from a right-to-left
 *  script, the same rule `dir="auto"` follows. A faded end must then sit on
 *  the left, so such text is cut with "…" instead, which follows the
 *  direction by itself. */
export function isRightToLeft(text: string) {
  const first = text.match(FIRST_LETTER)?.[0];
  return !!first && RTL_LETTER.test(first);
}
