"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import { StoreWaitlistEntry } from "@/lib/autogpt-server-api/types";
import { useSupabaseStore } from "@/lib/supabase/hooks/useSupabaseStore";
import { useToast } from "@/components/molecules/Toast/use-toast";
import BackendAPI from "@/lib/autogpt-server-api/client";
import { Check } from "lucide-react";

interface JoinWaitlistModalProps {
  waitlist: StoreWaitlistEntry;
  onClose: () => void;
}

export function JoinWaitlistModal({
  waitlist,
  onClose,
}: JoinWaitlistModalProps) {
  const { user } = useSupabaseStore();
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const { toast } = useToast();

  async function handleJoin() {
    setLoading(true);
    try {
      const api = new BackendAPI();
      await api.joinWaitlist(waitlist.waitlist_id, user ? undefined : email);

      setSuccess(true);
      toast({
        title: "You're on the list!",
        description: `We'll notify you when ${waitlist.name} is ready.`,
      });

      // Close after a short delay to show success state
      setTimeout(() => {
        onClose();
      }, 1500);
    } catch (error) {
      console.error("Error joining waitlist:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to join waitlist. Please try again.",
      });
    } finally {
      setLoading(false);
    }
  }

  if (success) {
    return (
      <Dialog open={true} onOpenChange={onClose}>
        <DialogContent className="sm:max-w-[400px]">
          <div className="flex flex-col items-center justify-center py-8">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-green-100 dark:bg-green-900">
              <Check className="h-8 w-8 text-green-600 dark:text-green-400" />
            </div>
            <DialogTitle className="mb-2 text-center text-xl">
              You&apos;re on the list!
            </DialogTitle>
            <DialogDescription className="text-center">
              We&apos;ll notify you when {waitlist.name} is ready.
            </DialogDescription>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[400px]">
        <DialogHeader>
          <DialogTitle>Join waitlist</DialogTitle>
          <DialogDescription>
            {user
              ? `Get notified when ${waitlist.name} is ready to use.`
              : `Enter your email to get notified when ${waitlist.name} is ready.`}
          </DialogDescription>
        </DialogHeader>

        <div className="py-4">
          {user ? (
            <div className="rounded-lg bg-neutral-50 p-4 dark:bg-neutral-800">
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                You&apos;ll be notified at:
              </p>
              <p className="mt-1 font-medium text-neutral-900 dark:text-neutral-100">
                {user.email}
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              <Label htmlFor="email">Email address</Label>
              <Input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          )}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleJoin}
            loading={loading}
            disabled={!user && !email}
            className="bg-neutral-800 text-white hover:bg-neutral-700 dark:bg-neutral-700 dark:hover:bg-neutral-600"
          >
            {user ? "Join waitlist" : "Join with email"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
