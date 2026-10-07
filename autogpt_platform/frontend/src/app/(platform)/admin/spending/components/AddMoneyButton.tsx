"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
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
        size="md"
        variant="primary"
        onClick={(e) => {
          e.stopPropagation();
          setIsAddMoneyDialogOpen(true);
        }}
      >
        Add Dollars
      </Button>

      {/* Add $$$ Dialog */}
      <Dialog
        title="Add Dollars"
        variant="compact"
        controlled={{
          isOpen: isAddMoneyDialogOpen,
          set: setIsAddMoneyDialogOpen,
        }}
      >
        <Dialog.Content>
          <Text variant="body" unmask={false} as="div" tone="muted">
            <div className="mb-2">
              <span className="font-medium">User:</span> {userEmail}
            </div>
            <div>
              <span className="font-medium">Current balance:</span> $
              {(currentBalance / 100).toFixed(2)}
            </div>
          </Text>

          <form action={handleApproveSubmit}>
            <input type="hidden" name="id" value={userId} />
            <input
              type="hidden"
              name="amount"
              value={Math.round(parseFloat(dollarAmount) * 100)}
            />

            <div className="grid gap-4 py-4">
              <Input
                id="dollarAmount"
                label="Amount (in dollars)"
                labelVariant="body-medium"
                size="md"
                wrapperClassName="mb-0"
                type="amount"
                amountPrefix="$"
                decimalCount={2}
                value={dollarAmount}
                onChange={(e) => setDollarAmount(e.target.value)}
                placeholder="0.00"
              />
            </div>

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Textarea
                  id="comments"
                  label="Comments (Optional)"
                  name="comments"
                  placeholder="Why are you adding dollars?"
                  defaultValue={defaultComments || "We love you!"}
                />
              </div>
            </div>

            <Dialog.Footer>
              <Button
                type="button"
                variant="outline"
                size="md"
                onClick={() => setIsAddMoneyDialogOpen(false)}
                disabled={isSubmitting}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="primary"
                size="md"
                disabled={isSubmitting}
              >
                {isSubmitting ? "Adding..." : "Add Dollars"}
              </Button>
            </Dialog.Footer>
          </form>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
