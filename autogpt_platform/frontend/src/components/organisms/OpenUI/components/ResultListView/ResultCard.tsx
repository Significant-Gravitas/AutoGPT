import type { z } from "zod/v4";
import type { resultOption } from "@/lib/openui/catalog-results";
import { Button } from "@/components/atoms/Button/Button";
import { ResultImage } from "./ResultImage";

interface Props {
  option: z.infer<typeof resultOption>;
  selected: boolean;
  disabled: boolean;
  onSelect: () => void;
}

export function ResultCard({ option, selected, disabled, onSelect }: Props) {
  return (
    <article
      className={`flex min-w-0 flex-col gap-3 rounded-xl border p-4 ${selected ? "border-primary bg-primary/5" : "border-border"}`}
    >
      {option.image && <ResultImage image={option.image} />}
      <h4 className="break-words font-semibold">{option.title}</h4>
      <p className="text-sm text-muted-foreground">{option.description}</p>
      <dl className="space-y-2 text-sm">
        {option.facts.map((fact, index) => (
          <div key={index}>
            <dt className="text-xs text-muted-foreground">{fact.label}</dt>
            <dd className="break-words">
              {fact.value}
              {fact.source && (
                <>
                  {" "}
                  ·{" "}
                  <a
                    href={fact.source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-primary underline"
                  >
                    {fact.source.label}
                  </a>
                </>
              )}
            </dd>
          </div>
        ))}
      </dl>
      {option.source && (
        <a
          href={option.source.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs text-primary underline"
        >
          {option.source.label}
        </a>
      )}
      <Button
        type="button"
        size="small"
        variant={selected ? "primary" : "secondary"}
        aria-pressed={selected}
        disabled={disabled}
        onClick={onSelect}
        className="mt-auto min-h-11 whitespace-normal"
      >
        {selected ? "Remove selection:" : "Choose"} {option.title}
      </Button>
    </article>
  );
}
