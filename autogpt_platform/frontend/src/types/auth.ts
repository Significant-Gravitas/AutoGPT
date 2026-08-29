import { z } from "zod";
import { AUTH_PASSWORD_MIN_LENGTH } from "@/lib/auth/password-policy";

const MIN_LENGTH_MESSAGE = `Password must contain at least ${AUTH_PASSWORD_MIN_LENGTH} characters`;

export type LoginProvider = "google" | "github" | "discord";

export const loginFormSchema = z.object({
  email: z
    .string()
    .email()
    .max(128, "Email must contain at most 128 characters")
    .trim(),
  password: z
    .string()
    .min(6, "Password must contain at least 6 characters")
    .max(64, "Password must contain at most 64 characters"),
});

export const signupFormSchema = z
  .object({
    email: z
      .string()
      .email()
      .max(128, "Email must contain at most 128 characters")
      .trim(),
    password: z
      .string()
      .min(AUTH_PASSWORD_MIN_LENGTH, MIN_LENGTH_MESSAGE)
      .max(64, "Password must contain at most 64 characters"),
    confirmPassword: z
      .string()
      .min(AUTH_PASSWORD_MIN_LENGTH, MIN_LENGTH_MESSAGE)
      .max(64, "Password must contain at most 64 characters"),
    agreeToTerms: z.boolean().refine((value) => value === true, {
      message: "You must agree to the Terms of Use and Privacy Policy",
    }),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
  });

export const sendEmailFormSchema = z.object({
  email: z
    .string()
    .email()
    .max(128, "Email must contain at most 128 characters")
    .trim(),
});

export const changePasswordFormSchema = z
  .object({
    password: z
      .string()
      .min(AUTH_PASSWORD_MIN_LENGTH, MIN_LENGTH_MESSAGE)
      .max(64, "Password must contain at most 64 characters"),
    confirmPassword: z
      .string()
      .min(AUTH_PASSWORD_MIN_LENGTH, MIN_LENGTH_MESSAGE)
      .max(64, "Password must contain at most 64 characters"),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
  });
