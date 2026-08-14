"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { NAME_CHIPS } from "../../helpers";
import { useNameStep } from "./useNameStep";

interface Props {
  onSubmit: (name: string) => void;
}

export function NameStep({ onSubmit }: Props) {
  const { custom, setCustom, trimmed, submitCustom } = useNameStep({
    onSubmit,
  });

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap gap-2.5">
        {NAME_CHIPS.map((chip) => (
          <button
            key={chip}
            type="button"
            onClick={() => onSubmit(chip)}
            className="rounded-full border border-zinc-200 bg-white px-5 py-2.5 text-sm font-medium text-zinc-800 transition-colors hover:border-purple-300 hover:bg-purple-50/40"
          >
            {chip}
          </button>
        ))}
      </div>
      <form onSubmit={submitCustom} className="flex items-end gap-2">
        <Input
          id="raise-name"
          label="Or type your own"
          hideLabel
          value={custom}
          onChange={(event) => setCustom(event.target.value)}
          placeholder="Type a name…"
          maxLength={100}
          wrapperClassName="flex-1"
        />
        <Button
          type="submit"
          variant="primary"
          disabled={!trimmed}
          className="rounded-full"
        >
          Name me
        </Button>
      </form>
    </div>
  );
}
