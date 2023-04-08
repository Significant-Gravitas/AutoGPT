import { TextField } from "@mui/material"
import styled from "styled-components"

const STextField = styled(TextField)`
  width: 100%;
  margin: 1rem;
  & input {
    color: var(--grey100) !important;
    background-color: var(--grey700) !important;
    border-radius: 0.5rem !important;
  }
`

const SearchInput = () => {
  return <STextField placeholder="Search" variant="outlined" />
}

export default SearchInput
