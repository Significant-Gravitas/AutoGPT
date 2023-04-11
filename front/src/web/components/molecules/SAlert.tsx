import colored from "@/style/colored"
import { Alert } from "@mui/material"
import styled from "styled-components"

const SAlert = colored(styled(Alert)`
  background-color: hsla(
    var(--color-hue),
    var(--color-saturation),
    var(--color-lightness),
    0.1
  ) !important;
  color: var(--color) !important;
  & svg {
    color: var(--color) !important;
  }
  & p {
    color: var(--text-color) !important;
  }
`)

export default SAlert
