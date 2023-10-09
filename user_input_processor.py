import pandas as pd
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.metrics.pairwise import cosine_similarity


class UserInputProcessor:
    def __init__(self, cleaned_data_file):
        self.merged_data = pd.read_csv(cleaned_data_file)
        self.model_name = 'distilbert-base-uncased'
        self.model = DistilBertModel.from_pretrained(self.model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

    def process_and_compare(self, user_input, column='OFFER'):
        # Clean user input
        cleaned_user_input = self._clean_text(user_input)

        # Tokenize and obtain embeddings for user input
        tokens = self.tokenizer(cleaned_user_input, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            user_embedding = self.model(**tokens).last_hidden_state.mean(dim=1).squeeze().numpy()

        # Convert string representations of embeddings to lists
        self.merged_data['EMBEDDINGS'] = self.merged_data['EMBEDDINGS'].apply(self._convert_string_to_list)

        # Calculate cosine similarity with existing embeddings
        existing_embeddings = self.merged_data['EMBEDDINGS'].to_list()
        similarities = cosine_similarity([user_embedding], existing_embeddings)[0]

        # Create a DataFrame to store results
        result_df = pd.DataFrame({
            'OFFER': self.merged_data['OFFER'],
            'Similarity Score': similarities
            #,
            #'RETAILER': self.merged_data['RETAILER'],
            #'BRAND': self.merged_data['BRAND'],
            #'PRODUCT_CATEGORY': self.merged_data['PRODUCT_CATEGORY']
        })

        # Sort by similarity score in descending order
        result_df = result_df.sort_values(by='Similarity Score', ascending=False)
        #result_df=result_df[:,0:5]
        return result_df
    def _convert_string_to_list(self, string_representation):
        try:
            # Extract values between '[' and ']' and convert to a list of floats
            return [float(val) for val in string_representation[string_representation.find('[')+1:string_representation.rfind(']')].split()]
        except (ValueError, AttributeError):
            # Handle the case where the conversion fails
            return []



    def _clean_text(self, text):
        # Your cleaning logic here
        return text.lower()  # Placeholder, replace with your actual cleaning logic
