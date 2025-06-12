"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { useRouter } from "next/navigation";
import { addDollars } from "@/app/(platform)/admin/spending/actions";
import useCredits from "@/hooks/useCredits";
import { LoadingSpinner } from "@/components/ui/loading";

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
  const [isAddMoneyDialogOpen, setIsAddMoneyDialogOpen] = useState(false);
  const [isAdding, setIsAdding] = useState(false);
  const [dollarAmount, setDollarAmount] = useState(
    defaultAmount ? Math.abs(defaultAmount / 100).toFixed(2) : "1.00",
  );

  const { formatCredits } = useCredits();

  const handleApproveSubmit = async (formData: FormData) => {
    setIsAdding(true);
    try {
      await addDollars(formData);
      setIsAddMoneyDialogOpen(false);
      router.refresh(); // Refresh the current route
    } catch (error) {
      console.error("Error adding dollars:", error);
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <>
      <Button
        size="sm"
        variant="outline"
        onClick={() => setIsAddMoneyDialogOpen(true)}
        disabled={isAdding}
      >
        {isAdding ? (
          <>
            <LoadingSpinner className="mr-2 h-4 w-4" />
            Adding...
          </>
        ) : (
          "Add Money"
        )}
      </Button>

      <Dialog open={isAddMoneyDialogOpen} onOpenChange={setIsAddMoneyDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Money to {userEmail}</DialogTitle>
            <DialogDescription>
              Current Balance: {formatCredits(currentBalance)}
              <br />
              New Balance:{" "}
              {formatCredits(
                currentBalance + Math.round(parseFloat(dollarAmount) * 100),
              )}
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
                  <Input
                    id="dollarAmount"
                    type="number"
                    step="0.01"
                    className="rounded-l-none"
                    value={dollarAmount}
                    onChange={(e) => setDollarAmount(e.target.value)}
                    placeholder="0.00"
                    disabled={isAdding}
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
                  disabled={isAdding}
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsAddMoneyDialogOpen(false)}
                disabled={isAdding}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isAdding}>
                {isAdding ? (
                  <>
                    <LoadingSpinner className="mr-2 h-4 w-4" />
                    Adding...
                  </>
                ) : (
                  "Add Dollars"
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}
