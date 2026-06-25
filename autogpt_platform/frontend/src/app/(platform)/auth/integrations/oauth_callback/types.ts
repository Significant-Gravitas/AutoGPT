export type OAuthPopupResultMessage = { message_type: "oauth_popup_result" } & (
  | {
      success: true;
      code: string;
      state: string;
    }
  | {
      success: false;
      // Optional — present when the callback received a `state` token even
      // though it could not produce a successful result. Listeners filter
      // payloads by state, so an error without it would be silently dropped.
      state?: string;
      message: string;
    }
);
