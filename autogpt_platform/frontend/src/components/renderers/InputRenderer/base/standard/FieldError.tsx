import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Text } from "@/components/atoms/Text/Text";

export const FieldError = ({
  nodeId,
  fieldId,
  id,
}: {
  nodeId: string;
  fieldId: string;
  id?: string;
}) => {
  const nodeErrors = useNodeStore((state) => {
    const node = state.nodes.find((n) => n.id === nodeId);
    return node?.data?.errors;
  });
  const fieldError =
    nodeErrors?.[fieldId] || nodeErrors?.[fieldId.replace(/_%_/g, ".")] || null;

  return (
    <div id={id} aria-live="polite" aria-atomic="true">
      {fieldError && (
        <Text variant="small" className="mt-1 pl-4 !text-red-600">
          {fieldError}
        </Text>
      )}
    </div>
  );
};
