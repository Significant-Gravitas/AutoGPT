import { ArrowExpand01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";

interface LinkProps {
  label: string;
  href: string;
  onClick?: never;
  pressed?: never;
}

interface ActionProps {
  label: string;
  href?: never;
  onClick: () => void;
  pressed?: boolean;
}

type Props = LinkProps | ActionProps;

export function HomeTileExpandButton(props: Props) {
  const content = (
    <Icon icon={ArrowExpand01Icon} size={16} aria-hidden="true" />
  );

  if (props.href) {
    return (
      <Button
        as="NextLink"
        href={props.href}
        variant="icon"
        size="icon"
        className="size-8 shrink-0 rounded-full p-0"
        aria-label={props.label}
      >
        {content}
      </Button>
    );
  }

  return (
    <Button
      variant="icon"
      size="icon"
      className="size-8 shrink-0 rounded-full p-0"
      aria-label={props.label}
      aria-pressed={props.pressed}
      onClick={props.onClick}
    >
      {content}
    </Button>
  );
}
