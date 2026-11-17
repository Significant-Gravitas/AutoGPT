import {
  SIGN_IN_CODE_EXPIRED,
  SIGN_IN_CODE_LOCKED,
  SIGN_IN_CODE_OUT_OF_TRIES,
  SIGN_IN_CODE_START_ERRORS,
  SIGN_IN_CODE_UNCHECKED,
} from "@/lib/auth/sign-in-code";
import type { SignInCodeEnd } from "../../helpers";

const SIGN_IN_AGAIN = "Sign in again and we'll email you a new code.";

export const ENDED_SIGN_IN_COPY: Record<
  SignInCodeEnd,
  { title: string; body: string }
> = {
  [SIGN_IN_CODE_EXPIRED]: {
    title: "This sign-in has expired",
    body: SIGN_IN_AGAIN,
  },
  [SIGN_IN_CODE_OUT_OF_TRIES]: {
    title: "Too many wrong codes",
    body: SIGN_IN_AGAIN,
  },
  // No new code would help: the address gets none for now.
  [SIGN_IN_CODE_LOCKED]: {
    title: "Codes are paused for this address",
    body: SIGN_IN_CODE_START_ERRORS[SIGN_IN_CODE_LOCKED],
  },
  [SIGN_IN_CODE_UNCHECKED]: {
    title: "Something went wrong",
    body: `We couldn't finish checking your code. ${SIGN_IN_AGAIN}`,
  },
};
