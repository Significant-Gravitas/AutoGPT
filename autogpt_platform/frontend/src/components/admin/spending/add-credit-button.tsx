"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { CheckCircle, XCircle } from "lucide-react";
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
import type { StoreSubmission } from "@/lib/autogpt-server-api/types";
import { useRouter } from "next/navigation";
import { approveAgent, rejectAgent } from "@/app/admin/marketplace/actions";
import { addDollars } from "@/app/admin/spending/actions";

export function AddCreditButton({
  userId,
  userEmail,
  currentBalance,
}: {
  userId: string;
  userEmail: string;
  currentBalance: number;
}) {
  const router = useRouter();
  const [isAddCreditDialogOpen, setIsAddCreditDialogOpen] = useState(false);

  const handleApproveSubmit = async (formData: FormData) => {
    setIsAddCreditDialogOpen(false);
    try {
      await addDollars(formData);
      router.refresh(); // Refresh the current route
    } catch (error) {
      console.error("Error approving agent:", error);
    }
  };

  return (
    <>
      <Button
        size="sm"
        variant="outline"
        className="text-green-600 hover:bg-green-50 hover:text-green-700"
        onClick={(e) => {
          e.stopPropagation();
          setIsAddCreditDialogOpen(true);
        }}
      >
        <CheckCircle className="mr-2 h-4 w-4" />
        Add Dollars
      </Button>


      {/* Add $$$ Dialog */}
      <Dialog open={isAddCreditDialogOpen} onOpenChange={setIsAddCreditDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Dollars</DialogTitle>
            <DialogDescription>
              Are you sure you want to add $$$ to this user?
              Current balance: {currentBalance}
              User Email: {userEmail}
            </DialogDescription>
          </DialogHeader>

          <form action={handleApproveSubmit}>
            <input
              type="hidden"
              name="id"
              value={userId}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="amount">Amount</Label>
                <Textarea
                  id="amount"
                  name="amount"
                  placeholder="Enter the amount of $$$ to add"
                  required
                />
              </div>
            </div>

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments (Optional)</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Why are you adding $$$?"
                  defaultValue="we love you"
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsAddCreditDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button type="submit">Add Dollars</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>


    </>
  );
}
