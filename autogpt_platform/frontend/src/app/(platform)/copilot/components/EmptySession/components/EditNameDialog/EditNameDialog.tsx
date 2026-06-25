"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { PencilSimpleIcon } from "@phosphor-icons/react";
import { useState } from "react";

interface Props {
  currentName: string;
}

export function EditNameDialog({ currentName }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [name, setName] = useState(currentName);
  const [isSaving, setIsSaving] = useState(false);
  const { refreshSession } = useSupabase();
  const { toast } = useToast();

  function handleOpenChange(open: boolean) {
    if (open) setName(currentName);
    setIsOpen(open);
  }

  async function handleSave() {
    const trimmed = name.trim();
    if (!trimmed) return;

    setIsSaving(true);
    try {
      const res = await fetch("/api/auth/user", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_name: trimmed }),
      });

      if (!res.ok) {
        const body = await res.json();
        toast({
          title: "Failed to update name",
          description: body.error ?? "Unknown error",
          variant: "destructive",
        });
        return;
      }

      const session = await refreshSession();
      if (session?.error) {
        toast({
          title: "Name saved, but session refresh failed",
          description: session.error,
          variant: "destructive",
        });
        setIsOpen(false);
        return;
      }
      setIsOpen(false);
      toast({ title: "Name updated" });
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <Dialog
      title="Edit display name"
      styling={{ maxWidth: "24rem" }}
      controlled={{ isOpen, set: handleOpenChange }}
    >
      <Dialog.Trigger>
        <button
          type="button"
          className="ml-1 inline-flex items-center text-violet-500 transition-colors hover:text-violet-700"
        >
          <PencilSimpleIcon size={16} />
        </button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4 px-1">
          <Input
            id="display-name"
            label="Display name"
            placeholder="Your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleSave();
              }
            }}
          />
          <Button
            variant="primary"
            onClick={handleSave}
            disabled={!name.trim() || isSaving}
            loading={isSaving}
          >
            Save
          </Button>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
