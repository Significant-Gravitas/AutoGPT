import Router from "./Router"
import LeftPanel from "./components/UI/organisms/LeftPanel"
import Flex from "./style/Flex"
import GlobalStyle from "./style/GlobalStyle"
import styled from "styled-components"

const AppContainer = styled.div`
  display: flex;
  height: 100vh;
  width: 100vw;
  background-color: var(--grey800);
`

function App() {
  return (
    <AppContainer>
      <GlobalStyle />
      <Flex fullWidth>
        <LeftPanel />
        <Router />
      </Flex>
    </AppContainer>
  )
}

export default App
