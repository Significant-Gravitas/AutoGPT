import Router from "./Router";
import LeftPanel from "./components/UI/organisms/LeftPanel";
import Flex from "./style/Flex";
import GlobalStyle from "./style/GlobalStyle";

function App() {
	return (
		<>
			<GlobalStyle />
			<Flex>
				<LeftPanel />
				<Router />
			</Flex>
		</>
	);
}

export default App;
