import ollama
import json

class KeywordsExtractor:
    def __init__(self, input_file, output_file, model='llama3.1:latest'):
        self.input_file = input_file
        self.output_file = output_file
        self.model = model
        self.full_text = ""
        self.topics = ""
        self.keywords = []

    def _load_data(self):
        """Reads the JSON file and joins text values."""
        with open(self.input_file, 'r') as f:
            data = json.load(f)
        self.full_text = " ".join([item['text'] for item in data])
        return self.full_text

    def _get_short_topics(self):
        """Identifies core topics to act as anchors for keyword extraction."""
        prompt = f"""
        From the text below, provide 2-3 topics about it. Use 1-2 words per topic.
        - Do not include descriptions or introductory text.

        Text:
        {self.full_text}
        """
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={'temperature': 0.1}
        )
        self.topics = response['response'].strip()
        return self.topics

    def _extract_keywords(self):
        """Extracts technical entities based on identified topics."""
        prompt = f"""
            ### ROLE
            You are a Technical Entity Extractor. Your task is to extract keywords ONLY if they are mentioned in the provided text segment and align with the identified topics.

            ### THEMATIC ANCHORS (Core Topics)
            "{self.topics}"

            ### TARGET SEGMENT
            "{self.full_text}"

            ### STRICT EXTRACTION RULES:
            1. **SOURCE LIMITATION:** Extract ONLY from the TARGET SEGMENT.
            2. **TOPIC ALIGNMENT:** Keywords must relate to the THEMATIC ANCHORS.
            3. **CONVERSATIONAL FILTER:** Ignore generic words like "video", "today", "welcome".

            ### OUTPUT FORMAT
            Return ONLY a valid JSON list of strings.
            ["keyword1", "keyword2"]
        """

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            format='json',
            options={'temperature': 0.1}
        )

        try:
            # Flatten or clean the list to ensure it's just strings
            self.keywords = json.loads(response['response'])
        except json.JSONDecodeError:
            self.keywords = []
        return self.keywords

    def _save_results(self):
        """Saves the extracted keywords in the requested format."""
        output_dict = {"keywords": self.keywords}
        with open(self.output_file, 'w') as f:
            json.dump(output_dict, f, indent=4)
        print(f"Successfully saved results to {self.output_file}")

    def run(self):
        """Executes the full pipeline."""
        self._load_data()
        self._get_short_topics()
        self._extract_keywords()
        self._save_results()
        return self.keywords

# --- Execution ---
if __name__ == "__main__":
    # Update these paths as needed
    extractor = KeywordsExtractor(
        input_file='test_subtitles.json', 
        output_file='keywords_output.json'
    )
    extractor.run()