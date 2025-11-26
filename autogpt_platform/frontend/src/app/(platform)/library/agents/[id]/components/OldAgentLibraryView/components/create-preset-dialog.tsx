"use client";

import React, { useState } from "react";
import { Button } from "@/components/__legacy__/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { Input } from "@/components/__legacy__/ui/input";
import { Textarea } from "@/components/__legacy__/ui/textarea";

interface CreatePresetDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (name: string, description: string) => Promise<void> | void;
}

export function CreatePresetDialog({
  open,
  onOpenChange,
  onConfirm,
}: CreatePresetDialogProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const handleSubmit = async () => {
    if (name.trim()) {
      await onConfirm(name.trim(), description.trim());
      setName("");
      setDescription("");
      onOpenChange(false);
    }
  };

  const handleCancel = () => {
    setName("");
    setDescription("");
    onOpenChange(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Create Preset</DialogTitle>
          <DialogDescription>
            Give your preset a name and description to help identify it later.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <label htmlFor="preset-name" className="text-sm font-medium">
              Name *
            </label>
            <Input
              id="preset-name"
              placeholder="Enter preset name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={handleKeyDown}
              autoFocus
            />
          </div>
          <div className="grid gap-2">
            <label htmlFor="preset-description" className="text-sm font-medium">
              Description
            </label>
            <Textarea
              id="preset-description"
              placeholder="Optional description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={3}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={!name.trim()}>
            Create Preset
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
