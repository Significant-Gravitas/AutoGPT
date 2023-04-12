import { TextFieldProps } from "@mui/material"
import STextField from "../atom/STextField"

const SearchInput = (props: TextFieldProps) => {
  return <STextField placeholder="Search" variant="outlined"{...props} />
}

export default SearchInput
