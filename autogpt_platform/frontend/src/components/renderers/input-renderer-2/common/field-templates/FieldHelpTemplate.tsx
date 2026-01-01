import { FieldHelpProps, helpId } from "@rjsf/utils";
import { RichHelp } from "@rjsf/core";
import { cn } from "@/lib/utils";

export default function FieldHelpTemplate(props: FieldHelpProps) {
  const { fieldPathId, help, uiSchema, registry, hasErrors } = props;
  if (!help) {
    return null;
  }

  return (
    <span
      className={cn("text-xs font-medium text-muted-foreground", {
        "text-destructive": hasErrors,
      })}
      id={helpId(fieldPathId)}
    >
      <RichHelp help={help} registry={registry} uiSchema={uiSchema} />
    </span>
  );
}
