"use client";

import { Delete02Icon, PauseIcon, PlayIcon } from "@hugeicons/core-free-icons";

import { formatLastUsed } from "@/app/(platform)/settings/api-keys/components/APIKeyList/helpers";
import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

import { describeTrust, platformOf, platformOption } from "../../helpers";
import { RevokeWorkloadDialog } from "../RevokeWorkloadDialog/RevokeWorkloadDialog";
import { UsageDialog } from "../UsageDialog/UsageDialog";
import { useWorkloadRow } from "./useWorkloadRow";

interface Props {
  identity: WorkloadIdentityInfo;
}

export function WorkloadRow({ identity }: Props) {
  const {
    usageOpen,
    setUsageOpen,
    revokeOpen,
    setRevokeOpen,
    isDisabled,
    isChangingStatus,
    toggleStatus,
  } = useWorkloadRow(identity);
  const platform = platformOption(platformOf(identity));

  return (
    <div className="flex flex-col gap-3 py-4 pl-4 pr-5 lg:flex-row lg:items-center lg:justify-between">
      <div className="flex min-w-0 items-start gap-3">
        <Icon
          icon={platform.icon}
          size={20}
          className="mt-0.5 shrink-0 text-zinc-700"
        />
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex items-center gap-2">
            <Text variant="body-medium" as="span" className="text-textBlack">
              {identity.name}
            </Text>
            {isDisabled && (
              <Badge variant="warning" size="small">
                Disabled
              </Badge>
            )}
          </div>
          <Text variant="small" as="span" className="break-all text-zinc-600">
            {platform.label} · {describeTrust(identity.claim_conditions)}
          </Text>
          <Text variant="small" as="span" className="text-zinc-500">
            {formatLastUsed(identity.last_used_at)}
          </Text>
        </div>
      </div>

      <div className="flex shrink-0 flex-wrap items-center gap-2">
        <Button
          variant="secondary"
          size="small"
          onClick={() => setUsageOpen(true)}
        >
          How to use
        </Button>
        <Button
          variant="ghost"
          size="small"
          aria-label={`${isDisabled ? "Enable" : "Disable"} ${identity.name}`}
          leftIcon={<Icon icon={isDisabled ? PlayIcon : PauseIcon} size={16} />}
          loading={isChangingStatus}
          onClick={toggleStatus}
        >
          {isDisabled ? "Enable" : "Disable"}
        </Button>
        <Button
          variant="ghost"
          size="small"
          aria-label={`Revoke ${identity.name}`}
          leftIcon={<Icon icon={Delete02Icon} size={16} />}
          onClick={() => setRevokeOpen(true)}
        >
          Revoke
        </Button>
      </div>

      <UsageDialog
        identity={identity}
        open={usageOpen}
        onOpenChange={setUsageOpen}
      />
      <RevokeWorkloadDialog
        identity={identity}
        open={revokeOpen}
        onOpenChange={setRevokeOpen}
      />
    </div>
  );
}
