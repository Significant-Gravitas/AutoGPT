import Image from "next/image";
import { useState } from "react";
import type { z } from "zod/v4";
import type { Comparison } from "@/lib/openui/catalog-connected";
import { cn } from "@/lib/utils";
import { Button } from "@/components/atoms/Button/Button";

interface Props {
  option: z.infer<typeof Comparison.props>["options"][number];
  index: number;
  selected: boolean;
  preview: boolean;
  disabled: boolean;
  onChoose: () => void;
  buttonID: string;
}

export function ComparisonCard({
  option,
  index,
  selected,
  preview,
  disabled,
  onChoose,
  buttonID,
}: Props) {
  const [imageFailed, setImageFailed] = useState(false);
  return (
    <article
      className={cn(
        "flex min-w-0 flex-col gap-3 rounded-2xl border border-border bg-background p-3 transition-colors motion-reduce:transition-none md:p-4",
        (selected || preview) &&
          "border-primary bg-primary/5 ring-1 ring-primary",
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-1 text-xs font-medium text-muted-foreground">
        <span className="whitespace-nowrap">
          OPTION {index === 0 ? "A" : "B"}
        </span>
        {selected && <span className="text-primary">Selected</span>}
      </div>
      {option.image &&
        /^https?:\/\//.test(option.image.url) &&
        !imageFailed && (
          <Image
            src={option.image.url}
            alt={option.image.alt}
            width={480}
            height={240}
            unoptimized
            className="h-32 w-full rounded-lg object-cover"
            onError={() => setImageFailed(true)}
          />
        )}
      <div>
        <h4 className="break-words text-base font-semibold text-foreground">
          {option.title}
        </h4>
        <p className="mt-1 hidden text-sm text-muted-foreground md:block">
          {option.description}
        </p>
      </div>
      <dl className="space-y-2 text-sm">
        {option.facts?.map((fact, factIndex) => (
          <div key={factIndex} className="flex flex-col gap-0.5">
            <dt className="text-xs text-muted-foreground">{fact.label}</dt>
            <dd className="break-words text-foreground">{fact.value}</dd>
          </div>
        ))}
      </dl>
      <details className="text-sm text-muted-foreground md:hidden">
        <summary className="min-h-11 cursor-pointer py-3 text-xs font-medium text-foreground">
          About this option
        </summary>
        <p className="pb-2">{option.description}</p>
      </details>
      {option.source && /^https?:\/\//.test(option.source.url) && (
        <a
          className="text-xs text-primary underline underline-offset-2"
          href={option.source.url}
          target="_blank"
          rel="noopener noreferrer"
        >
          {option.source.label}
        </a>
      )}
      <Button
        id={buttonID}
        type="button"
        size="small"
        variant={selected ? "primary" : "secondary"}
        className="mt-auto h-auto min-h-11 w-full min-w-0 whitespace-normal break-words text-xs md:text-sm"
        aria-pressed={selected}
        disabled={disabled}
        onClick={onChoose}
        unmask={false}
      >
        Choose {option.title}
      </Button>
    </article>
  );
}
