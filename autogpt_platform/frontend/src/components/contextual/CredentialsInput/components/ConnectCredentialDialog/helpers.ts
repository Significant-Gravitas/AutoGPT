export interface ExistingCredential {
  id: string;
  title: string;
  type: string;
}

/** Accounts the user already has that could satisfy the request without a
 *  new sign-in — an expert chat offers these before the connect methods. */
export interface ExistingCredentialsOffer {
  credentials: ExistingCredential[];
  /** Resolves true once the credential is usable; false keeps the dialog
   *  open so the user can retry or add a new one instead. */
  onUse: (credential: ExistingCredential) => Promise<boolean>;
  isPending: boolean;
  error: string | null;
  /** Why the accounts are offered. "grant" hands one to an expert that lacks
   *  it; "choose" picks which of the user's own accounts a chat runs on;
   *  "update" picks which account to sign in to again for wider access, so the
   *  sign-in upgrades that account instead of adding another beside it. The
   *  last two pre-select nothing, because the choice is the whole point. */
  purpose?: "grant" | "choose" | "update";
}
