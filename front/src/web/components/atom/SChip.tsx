import colored from "@/style/colored";
import { Chip } from "@mui/material";
import styled from "styled-components";

const SChip = colored(styled(Chip)`
    border-color: var(--color) !important;
    color: var(--color) !important;
`);

export default SChip;