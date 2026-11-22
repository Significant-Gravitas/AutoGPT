import { useState } from "react";
import { Alert } from "react-native";

import { closeAccount } from "@/lib/api/account";
import { signOut } from "@/lib/session";

import { closedCopy, deleteErrorCopy } from "../../helpers";

export function useDeleteAccountSheet(email: string) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function deleteAccount() {
    setIsDeleting(true);
    setError(null);
    try {
      const { erase_after } = await closeAccount(email);
      Alert.alert("Your account is closed", closedCopy(erase_after));
      void signOut();
    } catch (caught) {
      setError(deleteErrorCopy(caught));
      setIsDeleting(false);
    }
  }

  return { isDeleting, error, deleteAccount };
}
