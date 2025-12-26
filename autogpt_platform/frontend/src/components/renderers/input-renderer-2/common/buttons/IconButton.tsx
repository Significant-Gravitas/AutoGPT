import {
  FormContextType,
  IconButtonProps,
  RJSFSchema,
  StrictRJSFSchema,
  TranslatableString,
} from "@rjsf/utils";
import { ChevronDown, ChevronUp, Copy, Trash2 } from "lucide-react";
import type { VariantProps } from "class-variance-authority";

import { Button } from "@/components/atoms/Button/Button";
import { extendedButtonVariants } from "@/components/atoms/Button/helpers";
import { TrashIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

export type AutogptIconButtonProps<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
> = IconButtonProps<T, S, F> & VariantProps<typeof extendedButtonVariants>;

export default function IconButton<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: AutogptIconButtonProps<T, S, F>) {
  const { icon, iconType, className, uiSchema, registry, ...otherProps } =
    props;
  return (
    <Button
      size="icon"
      variant="secondary"
      className={cn("border border-zinc-200 p-1.5")}
      {...otherProps}
      type="button"
    >
      {icon}
    </Button>
  );
}

export function CopyButton<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: AutogptIconButtonProps<T, S, F>) {
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

export function MoveDownButton<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: AutogptIconButtonProps<T, S, F>) {
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

export function MoveUpButton<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: AutogptIconButtonProps<T, S, F>) {
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

export function RemoveButton<
  T = any,
  S extends StrictRJSFSchema = RJSFSchema,
  F extends FormContextType = any,
>(props: AutogptIconButtonProps<T, S, F>) {
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
