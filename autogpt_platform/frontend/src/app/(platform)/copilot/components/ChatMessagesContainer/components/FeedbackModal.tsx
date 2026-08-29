"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Textarea } from "@/components/ui/textarea";
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
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) handleClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="mx-auto w-[95%] space-y-4">
          <p className="text-sm text-muted-foreground">
            Your feedback helps us improve. Share details below.
          </p>
          <Textarea
            placeholder="Tell us what went wrong or could be improved..."
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={4}
            maxLength={2000}
            className="resize-none"
          />
          <div className="flex items-center justify-between">
            <p className="text-xs text-muted-foreground">
              {comment.length}/2000
            </p>
            <div className="flex gap-2">
              <Button variant="outline" size="small" onClick={handleClose}>
                Cancel
              </Button>
              <Button
                size="small"
                onClick={handleSubmit}
                disabled={!comment.trim()}
              >
                Submit feedback
              </Button>
            </div>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
