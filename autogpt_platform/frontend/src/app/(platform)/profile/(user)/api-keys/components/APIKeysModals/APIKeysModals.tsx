"use client";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { Label } from "@/components/__legacy__/ui/label";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import { useAPIkeysModals } from "./useAPIkeysModals";
import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";

export const APIKeysModals = () => {
  const {
    isCreating,
    handleCreateKey,
    handleCopyKey,
    setIsCreateOpen,
    setIsKeyDialogOpen,
    isCreateOpen,
    isKeyDialogOpen,
    keyState,
    setKeyState,
  } = useAPIkeysModals();

  return (
    <div className="mb-4 flex justify-end">
      <Dialog
        title="Create New API Key"
        controlled={{ isOpen: isCreateOpen, set: setIsCreateOpen }}
      >
        <Dialog.Trigger>
          <Button size="small">Create Key</Button>
        </Dialog.Trigger>
        <Dialog.Content>
          <Text variant="body" tone="muted">
            Create a new AutoGPT Platform API key
          </Text>
          <div className="grid gap-4 py-4">
            <Input
              id="name"
              label="Name"
              labelVariant="body-medium"
              size="small"
              value={keyState.newKeyName}
              onChange={(e) =>
                setKeyState((prev) => ({
                  ...prev,
                  newKeyName: e.target.value,
                }))
              }
              placeholder="My AutoGPT Platform API Key"
            />
            <Input
              id="description"
              label="Description (Optional)"
              labelVariant="body-medium"
              size="small"
              value={keyState.newKeyDescription}
              onChange={(e) =>
                setKeyState((prev) => ({
                  ...prev,
                  newKeyDescription: e.target.value,
                }))
              }
              placeholder="Used for..."
            />
            <div className="grid gap-2">
              <Label>Permissions</Label>
              {Object.values(APIKeyPermission).map((permission) => (
                <div className="flex items-center space-x-2" key={permission}>
                  <Checkbox
                    id={permission}
                    checked={keyState.selectedPermissions.includes(permission)}
                    onCheckedChange={(checked: boolean) => {
                      setKeyState((prev) => ({
                        ...prev,
                        selectedPermissions: checked
                          ? [...prev.selectedPermissions, permission]
                          : prev.selectedPermissions.filter(
                              (p) => p !== permission,
                            ),
                      }));
                    }}
                  />
                  <Label htmlFor={permission}>{permission}</Label>
                </div>
              ))}
            </div>
          </div>
          <Dialog.Footer>
            <Button
              variant="secondary"
              size="small"
              onClick={() => setIsCreateOpen(false)}
            >
              Cancel
            </Button>
            <Button
              size="small"
              onClick={handleCreateKey}
              disabled={isCreating}
            >
              Create
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>

      <Dialog
        title="AutoGPT Platform API Key Created"
        controlled={{ isOpen: isKeyDialogOpen, set: setIsKeyDialogOpen }}
      >
        <Dialog.Content>
          <Text variant="body" tone="muted" className="mb-4">
            Please copy your AutoGPT API key now. You won&apos;t be able to see
            it again!
          </Text>
          <div className="flex items-center space-x-2">
            <code className="ph-no-capture flex-1 rounded-md bg-secondary p-2 text-sm">
              {keyState.newApiKey}
            </code>
            <Button
              size="icon-sm"
              variant="secondary"
              aria-label="Copy API key"
              leadingIcon={Copy01Icon}
              onClick={handleCopyKey}
            />
          </div>
          <Dialog.Footer>
            <Button size="small" onClick={() => setIsKeyDialogOpen(false)}>
              Close
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </div>
  );
};
