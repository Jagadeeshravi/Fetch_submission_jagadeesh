import pandas as pd
import re
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

class DataProcessor:
    def __init__(self, offer_retailer_file, categories_file, brand_category_file):
        self.offer_retailer = pd.read_csv(offer_retailer_file)
        self.categories = pd.read_csv(categories_file)
        self.brand_category = pd.read_csv(brand_category_file)
        self.merged_data = self._merge_datasets()

    def _merge_datasets(self):
        m1 = pd.merge(self.brand_category, self.categories, left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY', how='left')
        brands_category_aggregated = m1.groupby('BRAND').agg({
            'BRAND_BELONGS_TO_CATEGORY': list,
            'RECEIPTS': 'sum',
            'CATEGORY_ID': list,
            'PRODUCT_CATEGORY': list,
            'IS_CHILD_CATEGORY_TO': list
        }).reset_index()
        merged_df = pd.merge(self.offer_retailer, brands_category_aggregated[brands_category_aggregated['BRAND'].notna()], on='BRAND', how='left')
        return merged_df

    def handle_missing_values(self, strategy='constant', fill_value='Unknown'):
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        # Handle missing values in the aggregated columns from brands_category
        for column in ['BRAND_BELONGS_TO_CATEGORY', 'RECEIPTS', 'CATEGORY_ID', 'PRODUCT_CATEGORY', 'IS_CHILD_CATEGORY_TO']:
            self.merged_data[column].fillna(fill_value, inplace=True)

    def clean_text(self, column):
        # Apply text cleaning to the specified column
        self.merged_data[column] = self.merged_data[column].apply(lambda x: self._clean_text(x) if pd.notna(x) else x)
        print(self.merged_data.head())
    def _clean_text(self, text):
        # Convert to lowercase if not NaN
        if pd.notna(text):
            # Check if the input is an array (result of aggregation)
            if isinstance(text, list):
                # Apply cleaning to each element in the list
                cleaned_text = [self._clean_text_single(element) for element in text]
                return cleaned_text
            else:
                return self._clean_text_single(text)
        else:
            return text

    def _clean_text_single(self, text):
        # Convert to lowercase
        text = str(text).lower()
        # Remove special characters, numbers, and extra whitespaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(e for e in text.split() if e.isalnum())
        return text
    def create_embeddings(self, columns_to_concat):

        # Concatenate selected columns into a single text column
        text_column = self.merged_data[columns_to_concat].astype(str).agg(' '.join, axis=1)

        # Load pre-trained DistilBERT model and tokenizer
        model_name = 'distilbert-base-uncased'
        model = DistilBertModel.from_pretrained(model_name)
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)

        # Tokenize and obtain embeddings
        embeddings = []
        for text in text_column:
            tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = model(**tokens)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

        self.merged_data['EMBEDDINGS'] = embeddings
        print(self.merged_data.head())

    def save_cleaned_data(self, filename='cleaned_data.csv'):
        # Save the cleaned and merged data to a CSV file
        self.merged_data.to_csv(filename, index=False)
# Example usage:
processor = DataProcessor('D:\jagadeesh\offer_retailer.csv', 'D:\jagadeesh\categories.csv', 'D:\jagadeesh\Brand_category.csv')
#processor.handle_missing_values()
processor.clean_text('OFFER')
processor.clean_text('RETAILER')
processor.clean_text('BRAND')
#processor.clean_text('PRODUCT_CATEGORY')
columns_to_concat = ['OFFER', 'BRAND', 'RETAILER', 'PRODUCT_CATEGORY', 'IS_CHILD_CATEGORY_TO']

offer_embeddings = processor.create_embeddings(columns_to_concat)
processor.save_cleaned_data('cleaned_data.csv')
