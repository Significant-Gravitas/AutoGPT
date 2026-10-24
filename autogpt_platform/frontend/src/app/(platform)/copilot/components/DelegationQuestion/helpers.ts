/** A picked chip and free words read as one reply: "Q4. Keep December as a
 *  stretch note." Either alone is the whole answer. */
export function composeAnswer(picked: string | null, text: string): string {
  const words = text.trim();
  if (picked && words) return `${picked}. ${words}`;
  return picked ?? words;
}
