import { Text } from "@/components/atoms/Text/Text";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

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
  // What someone does says more than "AI Expert" does, so on a card or a page
  // it rides the name line and takes that line's place. Compact sizes have no
  // room for it and keep it on the line below. The category pill belongs to
  // the card, not here, so the two never say the same thing twice.
  const titleOnNameLine = compact ? null : area;

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
        {titleOnNameLine ? (
          <Text
            as="span"
            variant="body"
            tone="secondary"
            unmask={false}
            className={cn("min-w-0 truncate", areaClassName)}
            title={titleOnNameLine}
          >
            {"•"} {titleOnNameLine}
          </Text>
        ) : null}
        {nameAccessory}
      </Container>
      {!area ? (
        <Text as="span" variant="small" tone="muted">
          {isOtto ? "Your personal Head of AI" : "AI Expert"}
        </Text>
      ) : null}
      {compact && area ? (
        <Text
          as="span"
          variant="small"
          tone="secondary"
          unmask={false}
          className={cn("truncate leading-4", areaClassName)}
          title={area}
        >
          {area}
        </Text>
      ) : null}
    </Container>
  );
}
