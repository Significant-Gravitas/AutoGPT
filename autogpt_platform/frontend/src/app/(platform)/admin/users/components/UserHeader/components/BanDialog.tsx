"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { UserBlock01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import {
  type AdminUser,
  BAN_DURATIONS,
  type BanDuration,
  formatDateTime,
  formatTimeUntil,
  getBanEnd,
  getDisplayName,
} from "../../../helpers";
import { useIPReport } from "../../../useIPReport";
import { useUserBan } from "../../../useUserBan";

interface Props {
  user: AdminUser;
}

const DURATION_OPTIONS = BAN_DURATIONS.map(({ value, label }) => ({
  value,
  label,
}));

export function BanDialog({ user }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [reason, setReason] = useState("");
  const [duration, setDuration] = useState<BanDuration>("permanent");
  const { banUser, isBanning } = useUserBan(user);
  const { report } = useIPReport(user.id);
  const related = report?.related_total ?? 0;
  const banEnd = getBanEnd(duration);

  function handleOpenChange(open: boolean) {
    setIsOpen(open);
    if (open) {
      setReason("");
      setDuration("permanent");
    }
  }

  return (
    <Dialog
      title={`Ban ${getDisplayName(user)}?`}
      styling={{ maxWidth: "480px" }}
      controlled={{ isOpen, set: handleOpenChange }}
    >
      <Dialog.Trigger>
        <Button variant="secondary" size="small" leadingIcon={UserBlock01Icon}>
          Ban user…
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" className="text-zinc-600">
            They&apos;re signed out and can&apos;t sign in again until the ban
            ends, but for up to 5 minutes they can still get a token that works
            for an hour. Their API keys, schedules and triggers keep running.
          </Text>
          {related > 0 ? (
            <Text variant="small" className="text-zinc-500">
              Only this account. To include the{" "}
              {related === 1
                ? "1 related account"
                : `${related.toLocaleString("en-GB")} related accounts`}
              , use Ban all in Related accounts.
            </Text>
          ) : null}
          {user.role === "admin" ? (
            <Alert variant="warning">
              <AlertDescription>This account is an admin.</AlertDescription>
            </Alert>
          ) : null}
          <Input
            id="ban-reason"
            label="Reason (required)"
            type="textarea"
            rows={3}
            placeholder="What they did"
            value={reason}
            onChange={(event) => setReason(event.target.value)}
            required
            wrapperClassName="mb-0"
          />
          <div className="flex flex-col gap-1">
            <Select
              id="ban-duration"
              label="Duration"
              value={duration}
              onValueChange={(value) => setDuration(value as BanDuration)}
              options={DURATION_OPTIONS}
              wrapperClassName="mb-0"
            />
            <Text variant="small" className="text-zinc-500">
              {banEnd
                ? `Ends ${formatDateTime(banEnd)} (${formatTimeUntil(banEnd)})`
                : "Doesn't end on its own."}
            </Text>
          </div>
        </div>
        <Dialog.Footer>
          <Button
            variant="secondary"
            size="small"
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            size="small"
            loading={isBanning}
            disabled={!reason.trim()}
            onClick={() =>
              banUser(
                { reason, duration },
                { onSuccess: () => setIsOpen(false) },
              )
            }
          >
            Ban user
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
