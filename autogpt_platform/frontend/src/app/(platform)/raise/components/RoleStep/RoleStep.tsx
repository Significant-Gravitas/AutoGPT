"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { cn } from "@/lib/utils";
import { bubbleClassFor } from "../ColorStep/helpers";
import { CUSTOM_ROLE_MAX_LENGTH, roleOptionsForSelection } from "./helpers";
import { useRoleStep } from "./useRoleStep";

interface Props {
  selectedRole: string | null;
  color: string | null;
  onPick: (roleId: string) => void;
}

export function RoleStep({ selectedRole, color, onPick }: Props) {
  const { custom, setCustom, trimmed, submitCustom } = useRoleStep({
    onPick,
  });
  const options = roleOptionsForSelection(selectedRole);

  return (
    <div className="flex flex-col items-end gap-4">
      <div
        role="group"
        aria-label="What the expert does"
        className="flex flex-wrap justify-end gap-2.5"
      >
        {options.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => onPick(option.id)}
            disabled={Boolean(selectedRole)}
            aria-pressed={selectedRole ? true : undefined}
            className={cn(
              "rounded-full border px-5 py-2.5 text-sm font-medium text-foreground transition-colors",
              selectedRole
                ? (bubbleClassFor(color) ?? "border-accent bg-accent/5")
                : "border-border bg-background hover:border-accent hover:bg-accent/5",
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
      {selectedRole ? null : (
        <span
          aria-hidden
          className="mr-4 text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground"
        >
          or
        </span>
      )}
      {selectedRole ? null : (
        <form onSubmit={submitCustom} className="flex items-center gap-2">
          <Input
            id="raise-role"
            label="Or type your own"
            hideLabel
            size="small"
            value={custom}
            onChange={(event) => setCustom(event.target.value)}
            placeholder="Type a role…"
            maxLength={CUSTOM_ROLE_MAX_LENGTH}
            wrapperClassName="mb-0 w-full max-w-[16rem] [&_input]:h-[2.625rem] [&_input]:py-3"
          />
          <Button
            type="submit"
            variant="primary"
            size="small"
            disabled={!trimmed}
            className="h-[2.625rem] rounded-xl py-3"
          >
            Add role
          </Button>
        </form>
      )}
    </div>
  );
}
