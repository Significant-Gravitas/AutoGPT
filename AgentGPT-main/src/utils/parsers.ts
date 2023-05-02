import { z } from "zod";
import { StructuredOutputParser } from "langchain/output_parsers";

/*
 * Parsers are used by LangChain to easily prompt for a given format and also parse outputs.
 * https://js.langchain.com/docs/modules/prompts/output_parsers/
 */

export const respondAction = "Respond";
export const actionParser = StructuredOutputParser.fromZodSchema(
  z.object({
    // Enum type currently not supported
    action: z
      .string()
      .describe(`The action to take, either 'Question' or '${respondAction}'`),
    arg: z.string().describe("The argument to the action"),
  })
);

export const tasksParser = StructuredOutputParser.fromZodSchema(
  z.array(z.string()).describe("A list of tasks to complete")
);
