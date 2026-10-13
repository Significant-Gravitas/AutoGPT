import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { ExecutionView as ExecutionSchema } from "@/lib/openui/catalog-artifacts";
import { InlineExecution } from "@/components/organisms/InlineExecution/InlineExecution";
import { useOpenUIDisabled } from "../interactionContext";

export function ExecutionView({
  props,
}: ComponentRenderProps<z.infer<typeof ExecutionSchema.props>>) {
  const disabled = useOpenUIDisabled();
  return <InlineExecution {...props} disabled={disabled} />;
}
