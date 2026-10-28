"use client";

// TODO(tokens): no role for these colours yet, so they stay raw:
// bg-gray-50 #F9FAFB, text-gray-500 #6B7280.

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/molecules/Dialog/components/DialogPrimitives";
import { Label } from "@/components/atoms/Label/Label";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import { BaseInput } from "@/components/atoms/Input/BaseInput";
import { useRouter } from "next/navigation";
import { addDollars } from "@/app/(platform)/admin/spending/actions";
import { useToast } from "@/components/molecules/Toast/use-toast";

export function AdminAddMoneyButton({
  userId,
  userEmail,
  currentBalance,
  defaultAmount,
  defaultComments,
}: {
  userId: string;
  userEmail: string;
  currentBalance: number;
  defaultAmount?: number;
  defaultComments?: string;
}) {
  const router = useRouter();
  const { toast } = useToast();
  const [isAddMoneyDialogOpen, setIsAddMoneyDialogOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [dollarAmount, setDollarAmount] = useState(
    defaultAmount ? Math.abs(defaultAmount / 100).toFixed(2) : "1.00",
  );

  const handleApproveSubmit = async (formData: FormData) => {
    setIsSubmitting(true);
    try {
      await addDollars(formData);
      setIsAddMoneyDialogOpen(false);
      toast({
        title: "Success",
        description: `Added $${dollarAmount} to ${userEmail}'s balance`,
      });
      router.refresh(); // Refresh the current route
    } catch (error) {
      console.error("Error adding dollars:", error);
      toast({
        title: "Error",
        description: "Failed to add dollars. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <Button
        size="legacy-sm"
        variant="legacy-default"
        onClick={(e) => {
          e.stopPropagation();
          setIsAddMoneyDialogOpen(true);
        }}
      >
        Add Dollars
      </Button>

      {/* Add $$$ Dialog */}
      <Dialog
        open={isAddMoneyDialogOpen}
        onOpenChange={setIsAddMoneyDialogOpen}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Dollars</DialogTitle>
            <DialogDescription className="pt-2">
              <div className="mb-2">
                <span className="font-medium">User:</span> {userEmail}
              </div>
              <div>
                <span className="font-medium">Current balance:</span> $
                {(currentBalance / 100).toFixed(2)}
              </div>
            </DialogDescription>
          </DialogHeader>

          <form action={handleApproveSubmit}>
            <input type="hidden" name="id" value={userId} />
            <input
              type="hidden"
              name="amount"
              value={Math.round(parseFloat(dollarAmount) * 100)}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="dollarAmount">Amount (in dollars)</Label>
                <div className="flex">
                  <div className="flex items-center justify-center rounded-l-md border border-r-0 bg-gray-50 px-3 text-gray-500">
                    $
                  </div>
                  <BaseInput
                    id="dollarAmount"
                    type="number"
                    step="0.01"
                    className="rounded-l-none"
                    value={dollarAmount}
                    onChange={(e) => setDollarAmount(e.target.value)}
                    placeholder="0.00"
                  />
                </div>
              </div>
            </div>

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments (Optional)</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Why are you adding dollars?"
                  defaultValue={defaultComments || "We love you!"}
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="legacy-outline"
                onClick={() => setIsAddMoneyDialogOpen(false)}
                disabled={isSubmitting}
              >
                Cancel
              </Button>
              <Button
                variant="legacy-default"
                type="submit"
                disabled={isSubmitting}
              >
                {isSubmitting ? "Adding..." : "Add Dollars"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}
