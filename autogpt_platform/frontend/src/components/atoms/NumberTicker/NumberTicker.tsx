import { cn } from "@/lib/utils";

interface Props {
  value: number;
  className?: string;
}

interface Part {
  key: string;
  char: string;
  isDigit: boolean;
}

// Keyed from the right, so the ones column keeps its identity when the
// number gains a digit and only the columns that changed remount.
function splitDigits(text: string): Part[] {
  const chars = Array.from(text);
  const parts: Part[] = [];
  let digitIndex = 0;
  let symbolIndex = 0;
  for (let i = chars.length - 1; i >= 0; i--) {
    const char = chars[i];
    const isDigit = char >= "0" && char <= "9";
    parts.unshift({
      key: isDigit ? `d${digitIndex++}` : `s${symbolIndex++}`,
      char,
      isDigit,
    });
  }
  return parts;
}

/**
 * A number whose changed digits drop in, column by column, like an odometer.
 * Screen readers get the plain value; the animated digits are hidden from them.
 */
export function NumberTicker({ value, className }: Props) {
  const formatted = Number.isFinite(value) ? value.toLocaleString() : "0";
  const parts = splitDigits(formatted);

  return (
    <span
      className={cn("inline-flex leading-none tabular-nums", className)}
      data-slot="number-ticker"
    >
      <span className="sr-only">{formatted}</span>
      <span aria-hidden="true" className="inline-flex overflow-hidden">
        {parts.map((part) =>
          part.isDigit ? (
            <span
              key={`${part.key}-${part.char}`}
              className="inline-block animate-number-in"
            >
              {part.char}
            </span>
          ) : (
            <span key={part.key} className="inline-block">
              {part.char}
            </span>
          ),
        )}
      </span>
    </span>
  );
}
