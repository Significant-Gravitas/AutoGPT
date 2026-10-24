// ai-elements/prompt-input still reads this path. DS3b part 3 (SECRT-3113)
// moves ai-elements onto molecules/InputGroup and deletes this file.
export {
  InputGroup,
  InputGroupButton,
  InputGroupTextarea,
} from "@/components/molecules/InputGroup/InputGroup";
