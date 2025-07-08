from typing import List, Optional
from venv import logger
from tqdm import tqdm
from datasets import load_dataset

from trie_memory import TrieMemory



class HuggingFaceDatasetIntegration:
    """
    PRESERVED: Integration class for HuggingFace datasets adapted for trie-based architecture.
    """
    
    def __init__(self, trie_memory: TrieMemory):
        self.trie_memory = trie_memory
        self.dataset_configs = [
            ("bavard/personachat_truecased", None, "persona_chat"),
            ("daily_dialog", None, "daily_dialog"),
            ("wikitext", "wikitext-2-raw-v1", "wikitext_2_raw"),
            ("SocialGrep/the-reddit-dataset-dataset", "comments", "posts"),
        ]
        
        self.dataset_text_columns = {
            'persona_chat': ['personality', 'history', 'candidates'],
            'daily_dialog': ['dialog'],
            'wikitext_2_raw': ['text'],
            'comments': ['body', 'permalink'],
            'posts': ['title', 'selftext'],
        }
        
        logger.info(f"Initialized HuggingFaceDatasetIntegration with TrieMemory and {len(self.dataset_configs)} datasets")

    def process_dataset(self, dataset_name: str, config: Optional[str], friendly_name: str, 
                       max_samples: int = 1000, num_workers: int = 4, predictor_instance=None, 
                       shuffle: bool = True):
        """
        PRESERVED: Original process_dataset method with adaptation for trie memory.
        """
        text_columns = self.dataset_text_columns.get(friendly_name, ['text'])
        processed_count = 0

        try:
            logger.info(f"Processing {friendly_name} dataset with progressive token prediction...")

            loading_params = {
                "split": "train", 
                "streaming": True, 
                "trust_remote_code": True
            }

            dataset = None
            try:
                logger.info(f"Attempting to load {dataset_name} with num_workers={num_workers}")

                if config:
                    try:
                        dataset = load_dataset(dataset_name, config, num_workers=num_workers, **loading_params)
                        logger.info(f"✅ Successfully loaded {dataset_name} with config={config}")
                    except Exception as config_error:
                        logger.warning(f"Could not load with config parameter: {config_error}")
                        try:
                            dataset = load_dataset(f"{dataset_name}/{config}", num_workers=num_workers, **loading_params)
                            logger.info(f"✅ Successfully loaded {dataset_name}/{config} as subdataset")
                        except Exception as subdataset_error:
                            logger.warning(f"Could not load subdataset: {subdataset_error}")
                            dataset = load_dataset(dataset_name, num_workers=num_workers, **loading_params)
                else:
                    dataset = load_dataset(dataset_name, num_workers=num_workers, **loading_params)
                    logger.info(f"✅ Successfully loaded {dataset_name}")

            except Exception as num_workers_error:
                logger.warning(f"num_workers not supported for {dataset_name}: {num_workers_error}")
                logger.info(f"Falling back to loading {dataset_name} without num_workers")

                if config:
                    try:
                        dataset = load_dataset(dataset_name, config, **loading_params)
                        logger.info(f"✅ Successfully loaded {dataset_name} with config={config}")
                    except Exception as config_error:
                        try:
                            dataset = load_dataset(f"{dataset_name}/{config}", **loading_params)
                            logger.info(f"✅ Successfully loaded {dataset_name}/{config} as subdataset")
                        except Exception as subdataset_error:
                            dataset = load_dataset(dataset_name, **loading_params)
                else:
                    dataset = load_dataset(dataset_name, **loading_params)
                    logger.info(f"✅ Successfully loaded {dataset_name}")

            if shuffle and dataset:
                try:
                    buffer_size = min(10000, max_samples * 5)
                    logger.info(f"Applying buffered shuffling with buffer_size={buffer_size}")
                    dataset = dataset.shuffle(buffer_size=buffer_size, seed=42)
                    logger.info(f"✅ Successfully applied shuffling to {friendly_name} dataset")
                except Exception as shuffle_error:
                    logger.warning(f"Could not apply shuffling to {friendly_name}: {shuffle_error}")

            if predictor_instance is None:
                logger.warning("No predictor instance provided - falling back to trie processing only")
                return self._process_dataset_trie_only(dataset, friendly_name, text_columns, max_samples)

            if not hasattr(predictor_instance, 'run_progressive_token_prediction'):
                logger.error("Predictor instance does not have run_progressive_token_prediction method")
                raise AttributeError("Predictor instance missing run_progressive_token_prediction method")

            from tqdm import tqdm
            progress_bar = tqdm(total=max_samples, desc=f"Processing {friendly_name} with Progressive Prediction", unit="samples")

            logger.info(f"Starting progressive token prediction processing for {friendly_name}")

            for item in dataset:
                if processed_count >= max_samples:
                    logger.info(f"Reached maximum samples limit ({max_samples}) for {friendly_name}")
                    break
                
                item_texts = []

                for column in text_columns:
                    if column in item:
                        text_data = item[column]
                        if isinstance(text_data, str) and text_data.strip():
                            item_texts.append(text_data.strip())
                            logger.debug(f"Extracted text from column '{column}': {len(text_data)} chars")
                        elif isinstance(text_data, list):
                            for text_item in text_data:
                                if isinstance(text_item, str) and text_item.strip():
                                    item_texts.append(text_item.strip())
                                    logger.debug(f"Extracted list item from column '{column}': {len(text_item)} chars")

                if 'permalink' in item and item['permalink']:
                    title = self._extract_title_from_permalink(item['permalink'])
                    if title and title.strip():
                        item_texts.append(title.strip())
                        logger.debug(f"Added title from permalink: {title}")

                texts_processed_for_item = 0
                for text in item_texts:
                    try:
                        prediction_results = predictor_instance.run_progressive_token_prediction(
                            text, verbose=False
                        )

                        logger.debug(f"Progressive prediction completed: {prediction_results.get('total_predictions', 0)} predictions")
                        texts_processed_for_item += 1

                    except Exception as prediction_error:
                        logger.warning(f"Progressive prediction failed for text (len={len(text)}): {prediction_error}")
                        try:
                            tokens = self._tokenize_text(text)
                            if tokens:
                                self.trie_memory.add_sequence(tokens)
                                texts_processed_for_item += 1
                                logger.debug(f"Fallback: Added sequence with {len(tokens)} tokens to trie")
                        except Exception as fallback_error:
                            logger.warning(f"Fallback processing also failed: {fallback_error}")

                if texts_processed_for_item > 0:
                    processed_count += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({"Processed": processed_count})

                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count}/{max_samples} samples from {friendly_name}")

            progress_bar.close()
            logger.info(f"Successfully processed {processed_count} samples from {friendly_name}")
            return processed_count

        except Exception as e:
            logger.error(f"Failed to process {friendly_name}: {e}")
            if 'progress_bar' in locals():
                progress_bar.close()
            return None

    def _process_dataset_trie_only(self, dataset, friendly_name: str, text_columns: list, max_samples: int):
        """
        PRESERVED: Fallback processing using only trie memory.
        """
        logger.info(f"Using trie-only processing for {friendly_name}")

        
        progress_bar = tqdm(total=max_samples, desc=f"Processing {friendly_name} (Trie Only)", unit="samples")

        processed_count = 0

        try:
            for item in dataset:
                if processed_count >= max_samples:
                    break

                item_texts = []

                for column in text_columns:
                    if column in item:
                        text_data = item[column]
                        if isinstance(text_data, str) and text_data.strip():
                            item_texts.append(text_data.strip())
                        elif isinstance(text_data, list):
                            for text_item in text_data:
                                if isinstance(text_item, str) and text_item.strip():
                                    item_texts.append(text_item.strip())

                if 'permalink' in item and item['permalink']:
                    title = self._extract_title_from_permalink(item['permalink'])
                    if title:
                        title_tokens = self._tokenize_text(title)
                        if title_tokens:
                            self.trie_memory.add_sequence(title_tokens)

                texts_processed_for_item = 0
                for text in item_texts:
                    if len(text.strip()) > 10:
                        tokens = self._tokenize_text(text)
                        if tokens:
                            self.trie_memory.add_sequence(tokens)
                            texts_processed_for_item += 1

                if texts_processed_for_item > 0:
                    processed_count += 1
                    progress_bar.update(1)

            progress_bar.close()
            logger.info(f"Completed trie-only processing: {processed_count} samples from {friendly_name}")
            return processed_count

        except Exception as e:
            logger.error(f"Error in trie-only processing: {e}")
            progress_bar.close()
            return 0

    def _tokenize_text(self, text: str) -> List[str]:
        """
        PRESERVED: Simple tokenization for fallback processing.
        """
        try:
            for punct in ['.', '?', '!', ',', ';', ':', '(', ')', '[', ']', '{', '}']:
                text = text.replace(punct, f' {punct} ')
            
            tokens = [token.strip() for token in text.split() if token.strip()]
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []

    def _extract_title_from_permalink(self, permalink: str) -> str:
        """PRESERVED: Original _extract_title_from_permalink method"""
        try:
            parts = permalink.split('/')
            if len(parts) >= 7:
                title_slug = parts[6]
                title = title_slug.replace('_', ' ').replace('-', ' ')
                logger.debug(f"Extracted title from permalink: {title}")
                return title
            return ""
        except Exception as e:
            logger.warning(f"Could not extract title from permalink {permalink}: {e}")
            return ""