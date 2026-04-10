import json
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from langdetect import detect

class Summarizer:
    """
    A utility class to extract and summarize text content from JSON subtitle files.

    This class parses a JSON file containing subtitle data, detects the language 
    of the text, and utilizes Latent Semantic Analysis (LSA) to generate a 
    concise summary of the content.

    Attributes:
        file_path (str): The path to the .json file containing subtitle data.
    """

    def __init__(self, file_path):
        """
        Initializes the Summarizer with a file path and ensures NLTK resources are ready.

        Args:
            file_path (str): The system path to the target JSON file.
        """
        self.file_path = file_path
        self._ensure_resources()

    def _ensure_resources(self):
        """Downloads necessary NLTK tokenizers if not already present."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')

    def _load_text(self):
        """
        Reads the JSON file and concatenates text values.

        Returns:
            str: A single string containing all combined text from the JSON.
        """
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return " ".join([item['text'] for item in data])

    def get_summary(self, sentence_count=2):
        """
        Performs language detection and generates a summary.

        Args:
            sentence_count (int): The number of sentences to include in the summary. 
                Defaults to 2.

        Returns:
            tuple: A collection of sumy.models.dom.Sentence objects representing the summary.
        """
        full_text = self._load_text()
        
        # Detect language (e.g., 'en', 'fr', 'es') for the tokenizer
        language = detect(full_text)
        
        parser = PlaintextParser.from_string(full_text, Tokenizer(language))
        summarizer = LsaSummarizer()
        
        return summarizer(parser.document, sentence_count)
    
if __name__ == "__main__":
    test_subtitles = "../subtitles/my_subtitles.json"
    # Initialize the class
    summarizer_instance = Summarizer(test_subtitles)
    
    # Run the summarization
    print("Extracting summary...")
    summary = summarizer_instance.get_summary(sentence_count=2)
    
    # Print the output
    print("\n--- Summary Results ---")
    for sentence in summary:
        print(f"• {sentence}")