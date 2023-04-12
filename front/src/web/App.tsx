import LeftPanel from "@/components/organisms/LeftPanel"
import styled from "styled-components"
import Router from "./Router"
import Flex from "./style/Flex"
import GlobalStyle from "./style/GlobalStyle"
import { persistor, store } from "@/redux/store"
import { Provider } from "react-redux"
import { PersistGate } from "redux-persist/integration/react"

const AppContainer = styled.div`
  display: flex;
  height: 100vh;
  width: 100vw;
  background-color: var(--grey800);
`

function App() {
	return (
		<Provider store={store}>
			<PersistGate persistor={persistor}>
				<AppContainer>
					<GlobalStyle />
					<Flex fullWidth>
						<LeftPanel />
						<Router />
					</Flex>
				</AppContainer>
			</PersistGate>
		</Provider>
	)
}

export default App
