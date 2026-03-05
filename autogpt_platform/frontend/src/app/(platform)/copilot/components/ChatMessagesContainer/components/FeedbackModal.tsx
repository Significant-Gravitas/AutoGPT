"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { useState } from "react";

interface Props {
  isOpen: boolean;
  onSubmit: (comment: string) => void;
  onCancel: () => void;
}

export function FeedbackModal({ isOpen, onSubmit, onCancel }: Props) {
  const [comment, setComment] = useState("");

  function handleSubmit() {
    if (!comment.trim()) return;
    onSubmit(comment);
    setComment("");
  }

  function handleClose() {
    onCancel();
    setComment("");
  }

  return (
    <Dialog
      title="What could have been better?"
      styling={{ maxWidth: "30rem" }}
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) handleClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="mx-auto w-[99%] space-y-4">
          <Input
            label="Your feedback helps us improve. Share details below."
            id="feedback-textarea"
            type="textarea"
            placeholder="Tell us what went wrong or could be improved..."
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={4}
            maxLength={2000}
            className="w-full resize-none"
          />
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-400">{comment.length}/2000</p>
            <div className="flex gap-2">
              <Button variant="ghost" onClick={handleClose}>
                Cancel
              </Button>
              <Button onClick={handleSubmit} disabled={!comment.trim()}>
                Submit feedback
              </Button>
            </div>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
