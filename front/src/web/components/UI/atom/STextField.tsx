import styled from "styled-components"
import { TextField } from "@mui/material"

const STextField = styled(TextField)`
  width: 100%;
  margin: 1rem;
  & input {
    min-height: 3rem !important;
    color: var(--grey100) !important;
    background-color: var(--grey700) !important;
    border-radius: 0.5rem !important;
  }
  & .Mui-selected {
    outline: none !important;
  }
`

export default STextField
