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
  /** Category colour for the title on the name line: a gradient fill with
   *  a soft glow in that colour (card and page sizes only). */
  areaColor?: string;
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
  areaColor,
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
      data-slot="expert-identity"
      className={cn(
        "flex min-w-0 flex-col text-left",
        compact ? "gap-0" : "gap-1",
      )}
    >
      <Container
        data-slot="expert-identity-name-line"
        className={cn(
          "flex min-w-0 gap-2",
          size === "page" && "flex-wrap gap-y-0",
          nameAlign === "baseline" ? "items-baseline" : "items-center",
        )}
      >
        <Text
          as={size === "page" ? "h1" : "span"}
          variant={compact ? "body-medium" : "lead-semibold"}
          tone="primary"
          data-slot="expert-identity-name"
          className={cn(
            "min-w-0",
            size === "page" ? "max-w-full break-words" : "truncate",
            compact && "leading-4.5",
            // TODO(tokens): SECRT-3152, no text role for 24px.
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
            data-slot="expert-identity-title"
            className={cn(
              "min-w-0",
              size === "page" ? "max-w-full break-words" : "truncate",
              size === "card" && "text-body-sm leading-inherit",
              areaClassName,
            )}
            title={titleOnNameLine}
          >
            {areaColor ? (
              <>
                {"• "}
                <span
                  data-slot="expert-identity-title-text"
                  className={cn(
                    "bg-clip-text font-medium text-transparent",
                    size === "card" && "text-body-md leading-inherit",
                    // TODO(tokens): SECRT-3152, no text role for 18px.
                    size === "page" && "text-lg",
                  )}
                  style={{
                    backgroundImage: `linear-gradient(90deg, color-mix(in srgb, ${areaColor} 60%, rgb(var(--color-primary))), color-mix(in srgb, ${areaColor} 85%, rgb(var(--color-primary))))`,
                    filter: `drop-shadow(0 0 6px color-mix(in srgb, ${areaColor} 45%, transparent))`,
                  }}
                >
                  {titleOnNameLine}
                </span>
              </>
            ) : (
              <>
                {"•"} {titleOnNameLine}
              </>
            )}
          </Text>
        ) : null}
        {nameAccessory}
      </Container>
      {!area ? (
        <Text
          as="span"
          variant="small"
          tone="muted"
          data-slot="expert-identity-subtitle"
        >
          {isOtto ? "Your personal Head of AI" : "AI Expert"}
        </Text>
      ) : null}
      {compact && area ? (
        <Text
          as="span"
          variant="small"
          tone="secondary"
          data-slot="expert-identity-area"
          className={cn("truncate leading-4", areaClassName)}
          title={area}
        >
          {area}
        </Text>
      ) : null}
    </Container>
  );
}
