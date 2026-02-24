import { customizeValidator } from "@rjsf/validator-ajv8";

export const customValidator = customizeValidator({
  // Currently we do not have frontend side validation - we are only doing backend side validation
  // If in future we need validation on frontend - then i will add more condition here like max length, min length, etc.
  customFormats: {
    "short-text": /.*/, // Accept any string
    "long-text": /.*/,
  },
  ajvOptionsOverrides: {
    strict: false,
    validateFormats: false,
  },
});
