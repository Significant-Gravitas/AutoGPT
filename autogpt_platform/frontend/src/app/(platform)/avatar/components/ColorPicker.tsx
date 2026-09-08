import { Text } from "@/components/atoms/Text/Text";
import { COLORS, type ColorId } from "@/components/molecules/BotAvatar/helpers";
import { cn } from "@/lib/utils";
import { SectionHeading } from "./SectionHeading";

interface Props {
  selected: ColorId;
  onSelect: (color: ColorId) => void;
}

export function ColorPicker({ selected, onSelect }: Props) {
  return (
    <section className="space-y-3">
      <SectionHeading title="Colour" hint="role at a glance" />
      <div
        role="radiogroup"
        aria-label="Colour"
        className="grid grid-cols-4 gap-2 sm:grid-cols-8"
      >
        {COLORS.map((color) => {
          const isSelected = color.id === selected;
          return (
            <button
              key={color.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              aria-label={color.label}
              onClick={() => onSelect(color.id)}
              className={cn(
                "flex flex-col items-center gap-2 rounded-2xlarge border bg-white p-3 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300",
                isSelected
                  ? "border-zinc-900 ring-1 ring-zinc-900"
                  : "border-zinc-200 hover:border-zinc-300",
              )}
            >
              <span
                className="size-10 rounded-full border-2"
                style={{ backgroundColor: color.body, borderColor: color.mid }}
              />
              <span className="flex flex-col text-center">
                <Text
                  variant="small-medium"
                  as="span"
                  className="text-zinc-900"
                >
                  {color.label}
                </Text>
                <Text variant="small" as="span" className="text-zinc-500">
                  {color.role}
                </Text>
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
