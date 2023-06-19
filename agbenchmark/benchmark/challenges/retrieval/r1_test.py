from ..Challenge import Challenge


class RetrievelChallenge(Challenge):
    """ Chanllenge for information-retrieval """
    def __init__(self, json_data):
        self.json_data = json_data
        assert self.json_data["category"] == "information-retrieval"

    @property
    def agent_input(self):
        return self.json_data["query"]

    def scoring(self, content):
        for should_contain_word in self.json_data["ground"]["should_contain"]:
            if should_contain_word not in content:
                return 0.
        
        for should_not_contain_word in self.json_data["ground"]["should_not_contain"]:
            if should_not_contain_word in content:
                return 0.
        return 1.

    def run(self, output_file):
        output = open(output_file).read().strip()

        score = self.scoring(output)

        return score