import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Text } from "@/components/atoms/Text/Text";

export const FieldError = ({
  nodeId,
  fieldId,
}: {
  nodeId: string;
  fieldId: string;
}) => {
  const nodeErrors = useNodeStore((state) => {
    const node = state.nodes.find((n) => n.id === nodeId);
    return node?.data?.errors;
  });
  const fieldError =
    nodeErrors?.[fieldId] || nodeErrors?.[fieldId.replace(/_%_/g, ".")] || null;

  return (
    <div>
      {fieldError && (
        <Text variant="small" className="mt-1 pl-4 !text-red-600">
          {fieldError}
        </Text>
      )}
    </div>
  );
};
