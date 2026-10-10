"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { updateNativePush, useNativePushState } from "./useNativePush";

export function NativePushControl() {
  const { user } = useAuth();
  const { available, enabled, busy, error } = useNativePushState();
  if (!available || !user) return null;
  return (
    <section
      aria-label="Push notifications"
      className="flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-4"
    >
      <div>
        <Text variant="body-medium">Stay in the loop</Text>
        <Text variant="small" tone="secondary">
          Get notified when a chat has an update or your team needs a response.
        </Text>
      </div>
      <Button
        variant="secondary"
        loading={busy}
        disabled={busy}
        onClick={() =>
          updateNativePush(user.id, enabled ? "disable" : "enable")
        }
      >
        {enabled ? "Turn off notifications" : "Enable notifications"}
      </Button>
      {error && (
        <Text variant="small" role="alert">
          {error}
        </Text>
      )}
    </section>
  );
}
