"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Input } from "@/components/atoms/Input/Input";
import type { StoreWaitlistEntry } from "@/app/api/__generated__/models/storeWaitlistEntry";
import { useSupabaseStore } from "@/lib/supabase/hooks/useSupabaseStore";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV2AddSelfToTheAgentWaitlist } from "@/app/api/__generated__/endpoints/store/store";
import { Check } from "@phosphor-icons/react";

interface JoinWaitlistModalProps {
  waitlist: StoreWaitlistEntry;
  onClose: () => void;
  onSuccess?: () => void;
}

export function JoinWaitlistModal({
  waitlist,
  onClose,
  onSuccess,
}: JoinWaitlistModalProps) {
  const { user } = useSupabaseStore();
  const [email, setEmail] = useState("");
  const [success, setSuccess] = useState(false);
  const { toast } = useToast();
  const joinWaitlistMutation = usePostV2AddSelfToTheAgentWaitlist();

  function handleJoin() {
    joinWaitlistMutation.mutate(
      {
        waitlistId: waitlist.waitlistId,
        data: { email: user ? undefined : email },
      },
      {
        onSuccess: (response) => {
          if (response.status === 200) {
            setSuccess(true);
            toast({
              title: "You're on the list!",
              description: `We'll notify you when ${waitlist.name} is ready.`,
            });

            // Close after a short delay to show success state
            setTimeout(() => {
              onSuccess?.();
              onClose();
            }, 1500);
          } else {
            toast({
              variant: "destructive",
              title: "Error",
              description: "Failed to join waitlist. Please try again.",
            });
          }
        },
        onError: () => {
          toast({
            variant: "destructive",
            title: "Error",
            description: "Failed to join waitlist. Please try again.",
          });
        },
      },
    );
  }

  if (success) {
    return (
      <Dialog
        title=""
        controlled={{
          isOpen: true,
          set: async (open) => {
            if (!open) onClose();
          },
        }}
        onClose={onClose}
        styling={{ maxWidth: "400px" }}
      >
        <Dialog.Content>
          <div className="flex flex-col items-center justify-center py-8">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-green-100 dark:bg-green-900">
              <Check
                className="h-8 w-8 text-green-600 dark:text-green-400"
                size={32}
                weight="bold"
              />
            </div>
            <h2 className="mb-2 text-center text-xl font-semibold">
              You&apos;re on the list!
            </h2>
            <p className="text-center text-zinc-500">
              We&apos;ll notify you when {waitlist.name} is ready.
            </p>
          </div>
        </Dialog.Content>
      </Dialog>
    );
  }

  return (
    <Dialog
      title="Join waitlist"
      controlled={{
        isOpen: true,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
      onClose={onClose}
      styling={{ maxWidth: "400px" }}
    >
      <Dialog.Content>
        <p className="mb-4 text-sm text-zinc-500">
          {user
            ? `Get notified when ${waitlist.name} is ready to use.`
            : `Enter your email to get notified when ${waitlist.name} is ready.`}
        </p>

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
            <Input
              id="email"
              label="Email address"
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          )}
        </div>

        <Dialog.Footer>
          <Button type="button" variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={handleJoin}
            loading={joinWaitlistMutation.isPending}
            disabled={!user && !email}
            className="bg-neutral-800 text-white hover:bg-neutral-700 dark:bg-neutral-700 dark:hover:bg-neutral-600"
          >
            {user ? "Join waitlist" : "Join with email"}
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
