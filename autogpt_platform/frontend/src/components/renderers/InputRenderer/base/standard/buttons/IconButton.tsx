import {
  FormContextType,
  IconButtonProps,
  RJSFSchema,
  StrictRJSFSchema,
  TranslatableString,
} from "@rjsf/utils";
import { ChevronDown, ChevronUp, Copy } from "lucide-react";
import type { VariantProps } from "class-variance-authority";

import { Button } from "@/components/atoms/Button/Button";
import { extendedButtonVariants } from "@/components/atoms/Button/helpers";
import { TrashIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";

export type AutogptIconButtonProps<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
> = IconButtonProps<T, S, F> & VariantProps<typeof extendedButtonVariants>;

export default function IconButton(props: AutogptIconButtonProps) {
  const {
    icon,
    className,
    uiSchema: _uiSchema,
    registry: _registry,
    iconType: _iconType,
    ...otherProps
  } = props;

  return (
    <Button
      size="icon"
      variant="secondary"
      className={cn(className, "w-fit border border-zinc-200 p-1.5 px-4")}
      {...otherProps}
      type="button"
    >
      {icon}
      <Text variant="body" className="ml-2">
        {" "}
        Remove Item{" "}
      </Text>
    </Button>
  );
}

export function CopyButton(props: AutogptIconButtonProps) {
  const {
    registry: { translateString },
  } = props;
  return (
    <IconButton
      title={translateString(TranslatableString.CopyButton)}
      {...props}
      icon={<Copy className="h-4 w-4" />}
    />
  );
}

export function MoveDownButton(props: AutogptIconButtonProps) {
  const {
    registry: { translateString },
  } = props;
  return (
    <IconButton
      title={translateString(TranslatableString.MoveDownButton)}
      {...props}
      icon={<ChevronDown className="h-4 w-4" />}
    />
  );
}

export function MoveUpButton(props: AutogptIconButtonProps) {
  const {
    registry: { translateString },
  } = props;
  return (
    <IconButton
      title={translateString(TranslatableString.MoveUpButton)}
      {...props}
      icon={<ChevronUp className="h-4 w-4" />}
    />
  );
}

export function RemoveButton(props: AutogptIconButtonProps) {
  const {
    registry: { translateString },
  } = props;
  return (
    <IconButton
      title={translateString(TranslatableString.RemoveButton)}
      {...props}
      className={"border-destructive"}
      icon={<TrashIcon size={16} className="!text-zinc-800" />}
    />
  );
}
