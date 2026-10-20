import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { DeleteAccountFlow } from "@/components/organisms/DeleteAccountFlow/DeleteAccountFlow";
import { useDeleteAccountFlowOpener } from "@/components/organisms/DeleteAccountFlow/useDeleteAccountFlowOpener";
import { UserRemove01Icon } from "@hugeicons/core-free-icons";

// Rendered inside the paywall's Dialog.Content, so the flow is a nested Radix
// dialog: it opens as the top layer over the force-open paywall, and Radix
// counts clicks and focus inside it as inside the paywall.
export function DeleteAccountButton() {
  const { isFlowOpen, openFlow, handleFlowOpenChange } =
    useDeleteAccountFlowOpener("paywall");

  return (
    <>
      <Button
        variant="ghost"
        size="small"
        onClick={openFlow}
        leftIcon={<Icon icon={UserRemove01Icon} size={16} />}
        className="bg-white/90 text-zinc-500 hover:text-zinc-700"
      >
        Delete account
      </Button>
      <DeleteAccountFlow
        source="paywall"
        open={isFlowOpen}
        onOpenChange={handleFlowOpenChange}
      />
    </>
  );
}
