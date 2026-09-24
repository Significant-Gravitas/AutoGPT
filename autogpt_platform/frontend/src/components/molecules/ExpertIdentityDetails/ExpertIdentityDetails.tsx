import { Text } from "@/components/atoms/Text/Text";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { ExpertAreaChip } from "./components/ExpertAreaChip";

interface Props {
  name: string;
  isOtto?: boolean;
  role?: string | null;
  jobTitle?: string | null;
  size?: "compact" | "card" | "page";
  nameAccessory?: ReactNode;
  /** Badges and icons ride the middle of the name; a second line of type
   *  sits on its baseline. */
  nameAlign?: "center" | "baseline";
  areaClassName?: string;
}

export function ExpertIdentityDetails({
  name,
  isOtto = false,
  role,
  jobTitle,
  size = "card",
  nameAccessory,
  nameAlign = "center",
  areaClassName,
}: Props) {
  const compact = size === "compact";
  const Container = compact ? "span" : "div";
  const area = jobTitle || (role ? getExpertRoleLabel(role) : null);

  return (
    <Container
      className={cn(
        "flex min-w-0 flex-col text-left",
        compact ? "gap-0" : "gap-1",
      )}
    >
      <Container
        className={cn(
          "flex min-w-0 gap-2",
          nameAlign === "baseline" ? "items-baseline" : "items-center",
        )}
      >
        <Text
          as={size === "page" ? "h1" : "span"}
          variant={compact ? "body-medium" : "lead-semibold"}
          tone="primary"
          unmask={false}
          className={cn(
            "min-w-0 truncate",
            compact && "leading-[1.125rem]",
            size === "page" && "text-2xl leading-8",
          )}
        >
          {name}
        </Text>
        {nameAccessory}
      </Container>
      {!area || (!isOtto && !compact) ? (
        <Text as="span" variant="small" tone="muted">
          {isOtto ? "Your personal Head of AI" : "AI Expert"}
        </Text>
      ) : null}
      {!compact && area ? (
        <ExpertAreaChip role={role ?? ""} label={area} />
      ) : area ? (
        <Text
          as="span"
          variant="small"
          tone="secondary"
          unmask={false}
          className={cn("truncate", compact && "leading-4", areaClassName)}
          title={area}
        >
          {area}
        </Text>
      ) : null}
    </Container>
  );
}
