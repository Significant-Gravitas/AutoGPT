import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { isKey } from "@/lib/keyboard";
import { Add01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import type { KeyboardEvent } from "react";

interface Props {
  onAdd: (name: string) => void;
}

export function AddColumn({ onAdd }: Props) {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState("");

  function close() {
    setIsEditing(false);
    setName("");
  }

  function handleKeyDown(
    e: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) {
    if (isKey(e, "Enter")) {
      e.preventDefault();
      onAdd(name);
      close();
    }
    if (isKey(e, "Escape")) close();
  }

  if (!isEditing) {
    return (
      <Button
        variant="ghost"
        size="small"
        className="w-[17rem] shrink-0 justify-start rounded-2xl text-zinc-600"
        onClick={() => setIsEditing(true)}
        leftIcon={<Icon icon={Add01Icon} size={16} />}
      >
        Add column
      </Button>
    );
  }

  return (
    <div className="w-[17rem] shrink-0 rounded-2xl bg-zinc-100/70 p-2">
      <Input
        id="task-board-new-column"
        label="Column name"
        hideLabel
        autoFocus
        size="small"
        placeholder="Column name"
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={handleKeyDown}
        onBlur={close}
      />
    </div>
  );
}
