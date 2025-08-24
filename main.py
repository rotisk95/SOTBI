"""
FIXED: Complete Enhanced Trie-Based Predictive Token Embedding System

CRITICAL FIXES APPLIED:
1. FIXED beam search to only use actual trie children as continuations (not random nodes)
2. REMOVED complex fallback logic that was creating fake continuations
3. PRESERVED all working simple prediction logic
4. RESTORED basic trie traversal principles

ACCOUNTABILITY: Only _collect_multi_source_candidates method significantly changed.
All other functionality preserved exactly as working.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import json
import random
import re
import sys
import gc
import os
from queue import Queue
import sys
import msgpack
import numpy as np
import logging
import lmdb
import pickle
import time
from typing import List, Dict, Tuple, Optional, Any
import hashlib
import string
from collections import defaultdict, deque
from dataclasses import dataclass
import concurrent.futures
import numpy as np

from hf_dataset_integration import HuggingFaceDatasetIntegration
from llama_interface import LlamaInterface
from predictive_system import PredictiveSystem

# Configure logging for execution transparency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ADDED: HuggingFace integration imports and dependencies
try:
    from datasets import load_dataset
    from concurrent.futures import ThreadPoolExecutor, as_completed
    HUGGINGFACE_AVAILABLE = True
    logger.info("HuggingFace datasets library available")
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("HuggingFace datasets library not available - dataset training will be disabled")


class UserInteraction:
    """
    PRESERVED: Interactive user interface with menu system for the trie-based learning system.
    """
    
    def __init__(self, db_path: str = "./trie_memory_test.lmdb"):
        """Initialize the interactive system with trie-based memory."""
        try:
            self.system = PredictiveSystem(db_path)
            self.llm_model = LlamaInterface()
            self.hf_integration = HuggingFaceDatasetIntegration(self.system.trie_memory) if HUGGINGFACE_AVAILABLE else None
            self.session_stats = {
                'interactions': 0,
                'predictions': 0,
                'dataset_samples': 0,
                'start_time': time.time()
            }
            self.topic_progress = {}
            self.covered_topics = set()
            self.acceptable_progress_level = 0.8
            
            logger.info("Initialized UserInteraction with trie-based system")
        except Exception as e:
            logger.error(f"Failed to initialize UserInteraction: {e}")
            raise

    def run(self):
        """PRESERVED: Main interactive loop with JSON training support added"""
        print("🚀 Starting Recursive Token Weaver System...")
        logger.info("Starting interactive user interface")

        try:
            while True:
                self.show_menu()
                choice = input("Enter your choice (1-7): ").strip()  # MODIFIED: Updated range

                if choice == "1":
                    self._handle_interactive_learning()
                elif choice == "2":
                    self._handle_curriculum_training()
                elif choice == "3":
                    self._handle_hf_training()
                elif choice == "4":
                    self._handle_statistics()
                elif choice == "5":
                    self._handle_save_load()
                elif choice == "6":
                    print("👋 Goodbye!")
                    break
                elif choice == "7":  # ADDED: New case for JSON training
                    self._handle_json_training_robust()
                else:
                    print("❌ Invalid choice. Please enter 1-7.")  # MODIFIED: Updated range

        except KeyboardInterrupt:
            print("\n👋 System shutdown initiated")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"❌ System error: {e}")
        finally:
            try:
                self.system.close()
                logger.info("System shutdown completed")
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

    def process_json_conversation_dataset_robust(self, json_file_path: str, max_samples: int = None, 
                                               batch_size: int = 100, shuffle: bool = True) -> Dict[str, Any]:
        """
        ACCOUNTABILITY: Robust JSON conversation dataset processor with error recovery.

        FIXES APPLIED:
        - Line-by-line processing to handle malformed entries
        - Data structure validation and transformation
        - Memory-efficient streaming for large files
        - Comprehensive error reporting with line numbers
        - Graceful recovery from individual entry failures

        Args:
            json_file_path: Path to JSON file containing conversation data
            max_samples: Maximum number of samples to process (None for all)
            batch_size: Number of samples to process before logging progress
            shuffle: Whether to shuffle valid entries before processing

        Returns:
            Dict containing processing statistics and results
        """
        logger.info(f"Starting robust JSON conversation dataset processing: {json_file_path}")

        processing_stats = {
            'total_lines': 0,
            'valid_conversations': 0,
            'processed_conversations': 0,
            'malformed_json_errors': 0,
            'structure_errors': 0,
            'processing_errors': 0,
            'total_tokens': 0,
            'average_reward': 0.0,
            'error_lines': [],
            'start_time': time.time()
        }

        valid_conversations = []

        try:
            logger.info("Starting line-by-line JSON parsing for error recovery...")

            # Step 1: Process file line by line for error recovery
            with open(json_file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    processing_stats['total_lines'] = line_num

                    # Progress logging for large files
                    if line_num % 10000 == 0:
                        logger.info(f"Scanning line {line_num}...")

                    line = line.strip()
                    if not line:
                        continue

                    # Remove trailing comma if present (common JSON error)
                    if line.endswith(','):
                        line = line[:-1]

                    # Skip array brackets if processing JSONL-style within array
                    if line in ['[', ']']:
                        continue
                    
                    try:
                        # Step 2: Parse individual JSON entry
                        entry = json.loads(line)

                        # Step 3: Validate and transform data structure
                        transformed_entry = self._transform_json_entry(entry, line_num)

                        if transformed_entry:
                            valid_conversations.append(transformed_entry)
                            processing_stats['valid_conversations'] += 1
                        else:
                            processing_stats['structure_errors'] += 1
                            processing_stats['error_lines'].append({
                                'line': line_num,
                                'error': 'Structure validation failed',
                                'content': line[:100] + '...' if len(line) > 100 else line
                            })

                    except json.JSONDecodeError as e:
                        processing_stats['malformed_json_errors'] += 1
                        processing_stats['error_lines'].append({
                            'line': line_num,
                            'error': f'JSON parse error: {str(e)}',
                            'content': line[:100] + '...' if len(line) > 100 else line
                        })
                        logger.debug(f"JSON parse error at line {line_num}: {str(e)}")
                        continue
                    
                    except Exception as e:
                        processing_stats['structure_errors'] += 1
                        processing_stats['error_lines'].append({
                            'line': line_num,
                            'error': f'Processing error: {str(e)}',
                            'content': line[:100] + '...' if len(line) > 100 else line
                        })
                        logger.debug(f"Processing error at line {line_num}: {str(e)}")
                        continue
                    
            logger.info(f"File scanning completed: {processing_stats['total_lines']} lines processed")
            logger.info(f"Valid conversations found: {processing_stats['valid_conversations']}")
            logger.info(f"JSON errors: {processing_stats['malformed_json_errors']}")
            logger.info(f"Structure errors: {processing_stats['structure_errors']}")

            # Step 4: Check if we have any valid conversations
            if not valid_conversations:
                error_msg = "No valid conversations found in dataset"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Step 5: Shuffle if requested
            if shuffle:
                import random
                random.shuffle(valid_conversations)
                logger.info("Valid conversations shuffled")

            # Step 6: Limit samples if specified
            if max_samples and max_samples < len(valid_conversations):
                valid_conversations = valid_conversations[:max_samples]
                logger.info(f"Limited dataset to {max_samples} samples")

            # Step 7: Process valid conversations for training
            total_reward = 0.0
            processed_count = 0

            logger.info(f"Starting training on {len(valid_conversations)} valid conversations...")

            for idx, conversation in enumerate(valid_conversations):
                try:
                    user_input = conversation['input']
                    ai_output = conversation['output']

                    # Step 8: Use progressive token prediction for training
                    full_conversation = f"{user_input} {ai_output}"

                    prediction_result = self.run_progressive_token_prediction(
                        full_conversation, verbose=False
                    )

                    # Extract training metrics
                    if prediction_result:
                        tokens_processed = prediction_result.get('total_predictions', 0)
                        avg_reward = prediction_result.get('average_reward', 0.5)

                        processing_stats['total_tokens'] += tokens_processed
                        total_reward += avg_reward
                        processed_count += 1

                        logger.debug(f"Processed conversation {idx}: {tokens_processed} tokens, "
                                   f"avg_reward: {avg_reward:.3f}")

                    # Step 9: Direct input-output relationship training
                    input_output_pair = f"Q: {user_input} A: {ai_output}"
                    direct_result = self.system.process_input(input_output_pair, 0.8)

                    logger.debug(f"Direct training on conversation {idx}: "
                               f"{len(direct_result.get('tokens', []))} tokens processed")

                    # Step 10: Progress logging
                    if (idx + 1) % batch_size == 0:
                        elapsed_time = time.time() - processing_stats['start_time']
                        rate = (idx + 1) / elapsed_time if elapsed_time > 0 else 0
                        logger.info(f"Training progress: {idx + 1}/{len(valid_conversations)} conversations "
                                  f"({rate:.1f} conv/sec)")

                except Exception as conv_error:
                    processing_stats['processing_errors'] += 1
                    logger.error(f"Error training on conversation {idx}: {conv_error}")
                    continue
                
            # Step 11: Calculate final statistics
            processing_stats['processed_conversations'] = processed_count
            processing_stats['average_reward'] = total_reward / processed_count if processed_count > 0 else 0.0
            processing_stats['end_time'] = time.time()
            processing_stats['total_time'] = processing_stats['end_time'] - processing_stats['start_time']

            logger.info(f"Robust JSON dataset processing completed successfully:")
            logger.info(f"  - Total lines scanned: {processing_stats['total_lines']}")
            logger.info(f"  - Valid conversations: {processing_stats['valid_conversations']}")
            logger.info(f"  - Processed for training: {processed_count}")
            logger.info(f"  - JSON errors: {processing_stats['malformed_json_errors']}")
            logger.info(f"  - Structure errors: {processing_stats['structure_errors']}")
            logger.info(f"  - Processing errors: {processing_stats['processing_errors']}")
            logger.info(f"  - Total tokens: {processing_stats['total_tokens']}")
            logger.info(f"  - Average reward: {processing_stats['average_reward']:.3f}")
            logger.info(f"  - Total time: {processing_stats['total_time']:.1f} seconds")

            return processing_stats

        except FileNotFoundError:
            error_msg = f"JSON file not found: {json_file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        except Exception as e:
            error_msg = f"Unexpected error in robust JSON processing: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _transform_json_entry(self, entry: Dict[str, Any], line_num: int) -> Optional[Dict[str, str]]:
        """
        ACCOUNTABILITY: Transform complex JSON structures into simple input-output format.

        ADDED: Data structure transformation logic to handle various JSON formats.

        Args:
            entry: Raw JSON entry from file
            line_num: Line number for error reporting

        Returns:
            Transformed entry with 'input' and 'output' strings, or None if invalid
        """
        try:
            # Case 1: Already in expected format (simple input/output strings)
            if (isinstance(entry, dict) and 
                'input' in entry and 'output' in entry and
                isinstance(entry['input'], str) and isinstance(entry['output'], str)):

                input_text = entry['input'].strip()
                output_text = entry['output'].strip()

                if input_text and output_text:
                    return {
                        'input': input_text,
                        'output': output_text,
                        'conversation_id': entry.get('conversation_id', f'line_{line_num}'),
                        'sequence_position': entry.get('sequence_position', line_num)
                    }

            # Case 2: Complex nested input structure (like the problematic line)
            elif isinstance(entry, dict) and 'input' in entry:
                input_obj = entry['input']

                # Try to extract meaningful text from complex input structure
                extracted_input = self._extract_text_from_complex_object(input_obj)

                # Look for output in various possible locations
                output_text = None
                if 'output' in entry and isinstance(entry['output'], str):
                    output_text = entry['output'].strip()
                elif 'response' in entry and isinstance(entry['response'], str):
                    output_text = entry['response'].strip()
                elif 'answer' in entry and isinstance(entry['answer'], str):
                    output_text = entry['answer'].strip()

                if extracted_input and output_text:
                    return {
                        'input': extracted_input,
                        'output': output_text,
                        'conversation_id': entry.get('conversation_id', f'line_{line_num}'),
                        'sequence_position': entry.get('sequence_position', line_num)
                    }

            # Case 3: Other conversation formats (question/answer, etc.)
            elif isinstance(entry, dict):
                # Try various field name combinations
                input_fields = ['question', 'user', 'human', 'prompt', 'query']
                output_fields = ['answer', 'assistant', 'ai', 'response', 'reply']

                input_text = None
                output_text = None

                for field in input_fields:
                    if field in entry and isinstance(entry[field], str):
                        input_text = entry[field].strip()
                        break
                    
                for field in output_fields:
                    if field in entry and isinstance(entry[field], str):
                        output_text = entry[field].strip()
                        break
                    
                if input_text and output_text:
                    return {
                        'input': input_text,
                        'output': output_text,
                        'conversation_id': entry.get('conversation_id', f'line_{line_num}'),
                        'sequence_position': entry.get('sequence_position', line_num)
                    }

            # If no valid transformation found
            logger.debug(f"Could not transform entry at line {line_num}: unsupported structure")
            return None

        except Exception as e:
            logger.debug(f"Error transforming entry at line {line_num}: {str(e)}")
            return None

    def _extract_text_from_complex_object(self, obj: Any) -> Optional[str]:
        """
        ACCOUNTABILITY: Extract meaningful text from complex nested objects.

        ADDED: Text extraction logic for handling complex data structures.

        Args:
            obj: Complex object to extract text from

        Returns:
            Extracted text string or None if no meaningful text found
        """
        try:
            if isinstance(obj, str):
                return obj.strip() if obj.strip() else None

            elif isinstance(obj, dict):
                # Look for common text fields
                text_fields = [
                    'text', 'content', 'message', 'transcription', 
                    'description', 'prompt', 'query', 'question'
                ]

                for field in text_fields:
                    if field in obj and isinstance(obj[field], str):
                        text = obj[field].strip()
                        if text:
                            return text

                # If no direct text field, try to construct from metadata
                if 'metadata' in obj and isinstance(obj['metadata'], dict):
                    metadata = obj['metadata']
                    if 'transcription' in metadata and isinstance(metadata['transcription'], str):
                        text = metadata['transcription'].strip()
                        if text:
                            return text

            elif isinstance(obj, list) and obj:
                # Try to extract from first list item
                return self._extract_text_from_complex_object(obj[0])

            return None

        except Exception as e:
            logger.debug(f"Error extracting text from complex object: {str(e)}")
            return None

    def _handle_json_training_robust(self):
        """
        ACCOUNTABILITY: Enhanced JSON training handler with robust error recovery.

        REPLACES: _handle_json_training() with robust version that handles malformed JSON.
        """
        print("\n📄 ROBUST JSON CONVERSATION DATASET TRAINING")
        print("Features:")
        print("• Line-by-line processing with error recovery")
        print("• Handles malformed JSON entries gracefully")
        print("• Data structure transformation for complex formats")
        print("• Comprehensive error reporting")
        print("• Memory-efficient processing for large files")

        try:
            # Get file path from user
            json_file_path = input("Enter path to JSON conversation file: ").strip()

            if not json_file_path:
                print("❌ No file path provided")
                return

            # Get processing parameters
            max_samples_input = input("Max samples to process (press Enter for all): ").strip()
            max_samples = int(max_samples_input) if max_samples_input else None

            batch_size_input = input("Batch size for progress reporting (default 100): ").strip()
            batch_size = int(batch_size_input) if batch_size_input else 100

            shuffle_input = input("Shuffle dataset? (y/n, default y): ").strip().lower()
            shuffle = shuffle_input != 'n'

            print(f"\n🚀 Starting robust JSON dataset processing...")
            print(f"File: {json_file_path}")
            print(f"Max samples: {max_samples or 'All'}")
            print(f"Batch size: {batch_size}")
            print(f"Shuffle: {shuffle}")
            print(f"Mode: Line-by-line with error recovery")

            # Process the dataset with robust error handling
            results = self.process_json_conversation_dataset_robust(
                json_file_path=json_file_path,
                max_samples=max_samples,
                batch_size=batch_size,
                shuffle=shuffle
            )

            # Display comprehensive results
            print(f"\n✅ ROBUST JSON TRAINING COMPLETED:")
            print(f"   📊 Total lines scanned: {results['total_lines']}")
            print(f"   ✅ Valid conversations: {results['valid_conversations']}")
            print(f"   🎯 Processed for training: {results['processed_conversations']}")
            print(f"   🔤 Total tokens processed: {results['total_tokens']}")
            print(f"   🏆 Average reward: {results['average_reward']:.3f}")
            print(f"   ⏱️  Total time: {results['total_time']:.1f} seconds")

            # Error summary
            total_errors = (results['malformed_json_errors'] + 
                           results['structure_errors'] + 
                           results['processing_errors'])

            if total_errors > 0:
                print(f"\n⚠️  ERROR SUMMARY:")
                print(f"   📝 Malformed JSON entries: {results['malformed_json_errors']}")
                print(f"   🔧 Structure errors: {results['structure_errors']}")
                print(f"   ❌ Processing errors: {results['processing_errors']}")
                print(f"   📊 Total errors: {total_errors}")

                success_rate = (results['valid_conversations'] / results['total_lines']) * 100
                print(f"   ✅ Success rate: {success_rate:.1f}%")

                # Show sample errors for debugging
                if results['error_lines']:
                    print(f"\n🔍 Sample error details:")
                    for i, error in enumerate(results['error_lines'][:3], 1):
                        print(f"   {i}. Line {error['line']}: {error['error']}")
                        print(f"      Content: {error['content']}")

                    if len(results['error_lines']) > 3:
                        print(f"   ... and {len(results['error_lines']) - 3} more errors")

            if results['total_time'] > 0:
                rate = results['processed_conversations'] / results['total_time']
                print(f"   ⚡ Processing rate: {rate:.1f} conversations/second")

            # Update session stats
            if hasattr(self, 'session_stats'):
                self.session_stats['dataset_samples'] += results['processed_conversations']
                self.session_stats['interactions'] += results['processed_conversations']

            logger.info("Robust JSON training session completed successfully")

        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            logger.error(f"File not found in robust JSON training: {e}")

        except ValueError as e:
            print(f"❌ Data validation error: {e}")
            logger.error(f"Validation error in robust JSON training: {e}")

        except Exception as e:
            print(f"❌ Training error: {e}")
            logger.error(f"Error in robust JSON training: {e}")


    def show_menu(self):
        """PRESERVED: Display interactive menu for user with JSON training option added"""
        print("\n" + "="*60)
        print("🧠 RECURSIVE TOKEN WEAVER - HYBRID LEARNING SYSTEM WITH ACTIVATION")
        print("="*60)
        print("1. Interactive Learning Mode (Real-time prediction with RL feedback)")
        print("2. Curriculum LLM Training (Structured topic-based learning)")
        print("3. Train on HuggingFace datasets")
        print("4. View learning statistics")
        print("5. Save/Load model")
        print("6. Exit")
        print("7. Train on JSON conversation dataset")  # ADDED: New option
        print("="*60)

    def _handle_curriculum_training(self):
        """Handle curriculum-based LLM training session"""
        print("\n🎓 CURRICULUM LLM TRAINING MODE")
        print("Features:")
        print("• Structured topic-based learning with LLM interaction")
        print("• Curriculum progress tracking")
        print("• Real-time prediction and feedback")
        print("• Beam search and scoring options")
        print("\nInitializing curriculum training session...")

        try:
            # Initialize curriculum attributes if they don't exist
            if not hasattr(self, 'topic_progress'):
                self.topic_progress = {}
            if not hasattr(self, 'covered_topics'):
                self.covered_topics = set()
            if not hasattr(self, 'acceptable_progress_level'):
                self.acceptable_progress_level = 1.0  # Default threshold

            # Check if curriculum methods exist
            if not hasattr(self, 'choose_from_curriculum'):
                print("⚠️  Warning: choose_from_curriculum method not found")
                print("Please ensure curriculum.json exists and curriculum methods are implemented")
                return

            if not hasattr(self, 'llm_model'):
                print("⚠️  Warning: LLM model not initialized")
                print("Please ensure LlamaInterface is properly configured")
                return

            # Run the adapted training session
            self.interactive_llm_training_session()

        except Exception as e:
            logger.error(f"Error in curriculum training: {e}")
            print(f"❌ Error in curriculum training: {e}")
            print("Returning to main menu...")

    def run_progressive_token_prediction(self, text: str, verbose: bool = True) -> Dict[str, Any]:
        """
        PRESERVED: Progressive token prediction method for HuggingFace integration compatibility.
        """
        try:
            logger.info(f"Running progressive token prediction for text: '{text[:50]}...'")
            
            tokens = self.system._tokenize(text)
            if not tokens:
                logger.warning("No tokens found in text")
                return {'total_predictions': 0, 'average_reward': 0.0}
            
            total_predictions = 0
            total_reward = 0.0
            
            for i in range(1, len(tokens)):
                context = tokens[:i]
                actual_next = tokens[i]
                
                try:
                    context_embedding = self.system.trie_memory.context_window.current_context_embedding
                    result = self.system.process_input(' '.join(context + [actual_next]), 0.5)
                    
                    query_sequence_embedding = result.get('query_sequence_embedding', None)
                    if context_embedding is None or query_sequence_embedding is None:
                        logger.warning(f"Missing embeddings for context: {' '.join(context)}")
                        continue
                    predicted_tokens, confidence = self.system.predict_continuation(' '.join(context), context_embedding, query_sequence_embedding)

                    reward = 0.0
                    if predicted_tokens and len(predicted_tokens) > 0:
                        if predicted_tokens[0] == actual_next:
                            reward = confidence * 1.0
                        else:
                            reward = confidence * 0.1
                    
                    
                    
                    total_predictions += 1
                    total_reward += reward
                    
                    if verbose:
                        logger.debug(f"Prediction {i}: context='{' '.join(context)}' actual='{actual_next}' "
                                   f"predicted='{predicted_tokens[0] if predicted_tokens else 'None'}' reward={reward:.3f}")
                        
                except Exception as pred_error:
                    logger.warning(f"Prediction failed at position {i}: {pred_error}")
                    self.system.process_input(' '.join(context + [actual_next]), 0.1)
                    total_predictions += 1
                    total_reward += 0.1
            
            average_reward = total_reward / total_predictions if total_predictions > 0 else 0.0
            
            result = {
                'total_predictions': total_predictions,
                'total_reward': total_reward,
                'average_reward': average_reward
            }
            
            logger.info(f"Progressive prediction completed: {total_predictions} predictions, avg reward: {average_reward:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in progressive token prediction: {e}")
            return {'total_predictions': 0, 'average_reward': 0.0}
        
    def choose_from_curriculum(self, curriculum_path='curriculum.json'):
        """Choose a topic from the curriculum that hasn't been covered yet."""
        try:
            # Load the JSON file
            with open(curriculum_path, 'r') as file:
                curriculum_data = json.load(file)

            # Track all available topics to choose from
            available_topics = []

            # Check if curriculum has the expected structure
            if "curriculum" in curriculum_data and isinstance(curriculum_data["curriculum"], list):
                # Process each level
                for level_data in curriculum_data["curriculum"]:
                    if "topic" in level_data:
                        main_topic = level_data["topic"]

                        # Check if this main topic has been covered
                        if main_topic not in self.covered_topics:
                            available_topics.append(main_topic)

                        # Process subtopics if available
                        if "subtopics" in level_data and isinstance(level_data["subtopics"], list):
                            for subtopic in level_data["subtopics"]:
                                # Handle subtopics in string format
                                if isinstance(subtopic, str) and subtopic not in self.covered_topics:
                                    available_topics.append(subtopic)
                                # Handle subtopics in dictionary format
                                elif isinstance(subtopic, dict) and "topic" in subtopic:
                                    subtopic_name = subtopic["topic"]
                                    if subtopic_name not in self.covered_topics:
                                        available_topics.append(subtopic_name)

            # If we have available topics, randomly select one
            if available_topics:
                selected_topic = random.choice(available_topics)
                print(f"\nSelected curriculum topic: {selected_topic}")
                return selected_topic
            else:
                print("\nAll curriculum topics have been covered!")
                return None

        except FileNotFoundError:
            print(f"Error: {curriculum_path} not found.")
        except json.JSONDecodeError:
            print(f"Error: {curriculum_path} contains invalid JSON.")
        except Exception as e:
            print(f"Error processing curriculum: {str(e)}")

        # Return None if any error occurs
        return None
    

    
        
    def interactive_llm_training_session(self, response="I am a student, I learn from conversations."):
        """
        ENHANCED: Run automated curriculum learning with WORKING feedback system.

        ACCOUNTABILITY CHANGES:
        1. ADDED: Prediction tracking for feedback targeting
        2. REPLACED: Old feedback processing with working feedback system  
        3. ADDED: Immediate feedback impact verification
        4. PRESERVED: All existing LLM interaction and curriculum logic
        5. ENHANCED: Clear feedback results display
        6. FIXED: Consistent token extraction for all feedback scenarios

        Now uses the working feedback system for immediate, visible impact.
        """
        import random

        print("\n=== Fractal Memory System - Automated Curriculum Learning with WORKING Feedback ===")
        print("Running automated LLM conversation session with IMMEDIATE feedback impact...")
        print("Press Ctrl+C to stop the session")

        # Track session stats with feedback tracking
        session_stats = {
            "topics_covered": 0,
            "interactions": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "feedback_impact_total": 0.0,  # ADDED: Track feedback impact
            "predictions_changed": 0,       # ADDED: Track prediction changes
            "start_time": time.time()
        }

        # Learning mode settings
        use_beam_search = False
        show_details = False

        # Question templates for curriculum topics (PRESERVED)
        question_templates = [
            "Teach me about {topic}",
            "Explain {topic} to me",
            "What should I know about {topic}?",
            "Help me understand {topic}",
            "Can you discuss {topic}?",
            "I want to learn about {topic}",
            "Please tell me about {topic}",
            "What are the key concepts in {topic}?",
            "Give me an overview of {topic}",
            "I'm curious about {topic}",
            "Can you break down {topic} for me?",
            "What's important to understand about {topic}?",
            "I need to learn about {topic}",
            "Could you explain the basics of {topic}?",
            "What would you like me to know about {topic}?",
            "What do you know about {topic}?",
            "How does {topic} work?",
            "What are the main ideas behind {topic}?",
            "Can you provide examples of {topic}?"
        ]

        try:
            while True:
                # Step 1: Get curriculum topic (PRESERVED)
                print("\nSelecting curriculum topic...")
                curriculum_topic = self.choose_from_curriculum("curriculum.json")

                if curriculum_topic:
                    print(f"Selected topic: {curriculum_topic}")
                    selected_template = random.choice(question_templates)
                    formatted_question = selected_template.format(topic=curriculum_topic)
                    prompt = formatted_question
                    print(f"Question: {formatted_question}")
                else:
                    print("Using general knowledge prompt (no uncovered topics found)")
                    general_templates = [
                        "Tell me something interesting about any topic.",
                        "Share some knowledge with me.",
                        "What's something fascinating you can teach me?",
                        "I'm ready to learn about anything interesting.",
                        "What would you like to teach me today?"
                    ]
                    general_question = random.choice(general_templates)
                    prompt = response + " " + general_question
                    curriculum_topic = "general knowledge"
                    print(f"General question: {general_question}")

                # Step 2: Get LLM response (PRESERVED)
                print(f"\nSending prompt to LLM about: {curriculum_topic}")
                print("Waiting for LLM response (this may take some time)...")

                try:
                    llm_input = self.llm_model.get_response(prompt)
                    input_length = len(llm_input)
                    print(f"Received LLM response ({input_length} chars)")

                    preview_length = min(200, input_length)
                    print(f"\nLLM: {llm_input[:preview_length]}..." if input_length > preview_length else f"\nLLM: {llm_input}")
                except Exception as e:
                    print(f"Error getting LLM response: {str(e)}")
                    llm_input = f"Let me tell you about {curriculum_topic}. This topic involves learning important concepts."
                    print(f"Using fallback response: {llm_input[:50]}...")

                # Step 3: Process LLM input (PRESERVED)
                print("\nProcessing LLM input with curriculum context...")
                try:
                    contextual_input = f"{prompt} {llm_input}"
                    context_embedding = self.system.trie_memory.context_window.current_context_embedding
                    result = self.system.process_input(contextual_input, 0.7)
                    print(f"✅ Processed: {len(result['tokens'])} tokens")

                except Exception as e:
                    print(f"Warning - Error processing LLM input: {str(e)}")

                # CORRECTED: Remove unnecessary token extraction - system tracks predictions internally
                # Only extract actual_tokens when we have corrections (better_response)
                ai_response = ""
                confidence = 0.0
                prediction_generated = False

                # Step 4: Generate AI student response WITH TRACKING (ENHANCED)
                print("Generating AI student response with feedback tracking...")
                try:
                    query_sequence_embedding = result.get('query_sequence_embedding', None)

                    # Generate prediction WITH tracking for feedback
                    continuation, confidence = self.system.predict_continuation(
                        prompt, context_embedding, query_sequence_embedding, use_beam_search=use_beam_search
                    )

                    if continuation:
                        # ADDED: Print raw student continuation tokens for visibility
                        print(f"🔍 Student's Raw Continuation: {continuation}")
                        print(f"📊 Continuation Length: {len(continuation)} tokens")

                        # FIXED: Extract tokens consistently
                        output_tokens = []
                        for i, token in enumerate(continuation):
                            if token in {'?', '!', '.', ',', ';', ':'} and output_tokens:
                                output_tokens[-1] += token  # attach to previous token
                            else:
                                output_tokens.append(token)

                        prediction_text = ' '.join(output_tokens)
                        print(f"AI Student: {prediction_text}")
                        print(f"Confidence: {confidence:.3f}")
                        ai_response = prediction_text
                        prediction_generated = True

                        # FIXED: Track prediction with extracted tokens
                        print(f"✅ Extracted {len(output_tokens)} tokens for feedback tracking")

                    else:
                        # FIXED: Handle fallback case with proper token extraction
                        ai_response = f"I'm learning about {curriculum_topic}. This is very interesting and I want to understand more about these concepts."
                        print(f"AI Student (fallback): {ai_response}")

                        # FIXED: Extract tokens from fallback response
                        output_tokens = self.system._tokenize(ai_response)
                        print(f"✅ Extracted {len(output_tokens)} fallback tokens for feedback tracking")
                        prediction_generated = False

                    session_stats["interactions"] += 1

                except Exception as e:
                    print(f"Error generating response: {str(e)}")
                    # FIXED: Extract tokens from error fallback
                    ai_response = f"I'm studying {curriculum_topic}. It's fascinating to learn about these ideas."
                    print(f"AI Student (fallback): {ai_response}")
                    output_tokens = self.system._tokenize(ai_response)
                    print(f"✅ Extracted {len(output_tokens)} error fallback tokens for feedback tracking")
                    prediction_generated = False

                # FIXED: Ensure we always have tokens before proceeding
                if not output_tokens:
                    print("⚠️ Warning: No tokens extracted, creating minimal tokens")
                    output_tokens = self.system._tokenize(ai_response) if ai_response else ["<unknown>"]

                print(f"📊 Token extraction summary: {len(output_tokens)} tokens from {'prediction' if prediction_generated else 'fallback'}")

                # Step 5: Get LLM feedback (PRESERVED)
                print("\n🤖 Getting LLM feedback on AI student response...")
                try:
                    feedback_prompt = f"""
    Please evaluate the AI student's response and provide feedback in JSON format.

    CONTEXT:
    Topic: {curriculum_topic}
    Teacher input: {llm_input}

    AI STUDENT RESPONSE:
    {ai_response}

    Please provide your evaluation in the following JSON format:
    {{
        "score": <float between -1.0 and 1.0>,
        "reasoning": "<brief explanation of the score>",
        "better_response": "<optional: a better response the AI student could have given>",
        "specific_feedback": "<what the AI student did well or could improve>"
    }}

    Scoring guidelines:
    - 1.0: Excellent response, accurate, relevant, thoughtful
    - 0.5: Good response, mostly accurate and relevant
    - 0.0: Average response, some relevance but room for improvement
    - -0.5: Poor response, inaccurate or irrelevant content
    - -1.0: Very poor response, completely wrong or inappropriate

    RESPOND ONLY WITH VALID JSON, NO OTHER TEXT.
    """

                    llm_feedback_raw = self.llm_model.get_response(feedback_prompt)
                    print(f"Raw LLM feedback: {llm_feedback_raw[:200]}...")

                    # Parse JSON feedback (PRESERVED)
                    try:
                        import json
                        clean_feedback = llm_feedback_raw.strip()
                        if clean_feedback.startswith('```json'):
                            clean_feedback = clean_feedback[7:]
                        if clean_feedback.endswith('```'):
                            clean_feedback = clean_feedback[:-3]
                        clean_feedback = clean_feedback.strip()

                        # Fix common JSON issues
                        if not clean_feedback.endswith('}'):
                            lines = clean_feedback.split('\n')
                            valid_lines = []
                            for line in lines:
                                if ':' in line and not line.strip().endswith(','):
                                    if not line.strip().endswith(',') and not line.strip().endswith('}'):
                                        line = line.rstrip() + ','
                                valid_lines.append(line)
                            clean_feedback = '\n'.join(valid_lines)
                            if not clean_feedback.endswith('}'):
                                clean_feedback += '\n}'

                        llm_feedback = json.loads(clean_feedback)

                        # Extract feedback components
                        feedback_score = float(llm_feedback.get('score', 0.0))
                        feedback_score = max(-1.0, min(1.0, feedback_score))
                        reasoning = llm_feedback.get('reasoning', 'No reasoning provided')
                        better_response = llm_feedback.get('better_response', '')
                        specific_feedback = llm_feedback.get('specific_feedback', '')

                        print(f"\n📊 LLM Feedback:")
                        print(f"   Score: {feedback_score:.2f}")
                        print(f"   Reasoning: {reasoning}")
                        if specific_feedback:
                            print(f"   Specific Feedback: {specific_feedback}")
                        if better_response:
                            print(f"   Better Response: {better_response[:100]}...")

                        # ENHANCED: Apply feedback using WORKING feedback system with CORRECT token usage
                        print(f"\n🔧 Applying WORKING feedback system...")

                        # Store original prediction for comparison
                        original_ai_response = ai_response
                        original_confidence = confidence

                        # CORRECTED: Extract actual_tokens from better_response (if provided)
                        actual_tokens = None
                        if better_response:
                            # Extract tokens from LLM's better response suggestion
                            actual_tokens = self.system._tokenize(better_response)
                            print(f"🎯 Extracted {len(actual_tokens)} actual tokens from better_response: {actual_tokens[:5]}..." if len(actual_tokens) > 5 else f"🎯 Actual tokens: {actual_tokens}")
                        else:
                            print(f"🎯 No better_response provided - using score-only feedback")

                        # CORRECTED: Apply feedback with proper token distinction
                        # predicted_tokens are tracked internally via system.predict_continuation()
                        # actual_tokens come from better_response (what should have been predicted)
                        feedback_results = self.system.apply_prediction_feedback(
                            feedback_score, 
                            better_response if better_response else None, 
                            actual_tokens  # CORRECTED: Only send actual tokens (user correction)
                        )

                        # ADDED: Display immediate feedback impact
                        if 'error' not in feedback_results:
                            print(f"✅ WORKING FEEDBACK APPLIED:")
                            print(f"   🎯 Type: {feedback_results['feedback_type'].upper()}")
                            print(f"   📊 Nodes affected: {feedback_results['nodes_affected']}")
                            print(f"   💪 Total impact: {feedback_results['total_impact']:.3f}")
                            print(f"   🔧 Tokens processed: {len(output_tokens)}")

                            # Track feedback impact
                            session_stats["feedback_impact_total"] += feedback_results['total_impact']

                            # Show confidence changes
                            if feedback_results.get('confidence_changes'):
                                print(f"   🎯 CONFIDENCE CHANGES:")
                                for change in feedback_results['confidence_changes'][:3]:
                                    print(f"      '{change['token']}': {change['before']:.3f} → {change['after']:.3f} ({change['change']:+.3f})")

                            if feedback_results.get('correction_learned'):
                                print(f"   📚 Better response learned and reinforced")
                        else:
                            print(f"❌ Feedback error: {feedback_results['error']}")

                        # ADDED: Test feedback impact by generating new prediction
                        print(f"\n🔄 Testing feedback impact on future predictions...")
                        try:
                            test_continuation, test_confidence = self.system.predict_continuation(
                                contextual_input, context_embedding, query_sequence_embedding, use_beam_search=use_beam_search
                            )

                            if test_continuation:
                                test_output_tokens = []
                                for i, token in enumerate(test_continuation):
                                    if token in {'?', '!', '.', ',', ';', ':'} and test_output_tokens:
                                        test_output_tokens[-1] += token  # attach to previous token
                                    else:
                                        test_output_tokens.append(token)

                                new_ai_response = ' '.join(test_output_tokens)
                                print(f"🔮 NEW PREDICTION AFTER FEEDBACK:")
                                print(f"   Original: '{original_ai_response[:100]}{'...' if len(original_ai_response) > 100 else ''}'")
                                print(f"   New: '{new_ai_response[:100]}{'...' if len(new_ai_response) > 100 else ''}'")
                                print(f"   Confidence: {original_confidence:.3f} → {test_confidence:.3f} ({test_confidence - original_confidence:+.3f})")

                                # Check if prediction actually changed
                                if new_ai_response != original_ai_response:
                                    print(f"🎉 SUCCESS: AI response CHANGED due to feedback!")
                                    session_stats["predictions_changed"] += 1
                                elif abs(test_confidence - original_confidence) > 0.1:
                                    print(f"📈 PARTIAL: Confidence significantly changed")
                                else:
                                    print(f"📊 MINIMAL: Limited visible change (feedback still applied)")
                        except Exception as test_error:
                            print(f"⚠️ Error testing feedback impact: {test_error}")

                        # Update session stats
                        if feedback_score > 0.0:
                            session_stats["positive_feedback"] += 1
                        else:
                            session_stats["negative_feedback"] += 1

                        print(f"✅ Applied LLM feedback with WORKING system: {feedback_score:.2f}")

                    except json.JSONDecodeError as e:
                        print(f"❌ Error parsing LLM feedback JSON: {str(e)}")
                        print(f"Raw response: {llm_feedback_raw}")

                        # FIXED: Manual score extraction fallback with proper tokens
                        try:
                            import re
                            score_match = re.search(r'"score":\s*([0-9.-]+)', llm_feedback_raw)
                            if score_match:
                                manual_score = float(score_match.group(1))
                                manual_score = max(-1.0, min(1.0, manual_score))
                                print(f"🔧 Extracted score manually: {manual_score}")

                                # CORRECTED: Apply manual feedback using working system (score-only)
                                print(f"🎯 Applying manual feedback (score-only, no correction)")
                                manual_results = self.system.apply_prediction_feedback(
                                    manual_score, 
                                    None, 
                                    None  # CORRECTED: No actual_tokens for score-only feedback
                                )
                                print(f"✅ Applied manual feedback: impact {manual_results.get('total_impact', 0.0):.3f}")
                                session_stats["feedback_impact_total"] += manual_results.get('total_impact', 0.0)

                                if manual_score > 0.0:
                                    session_stats["positive_feedback"] += 1
                                else:
                                    session_stats["negative_feedback"] += 1
                            else:
                                # CORRECTED: Neutral feedback fallback (score-only)
                                print(f"🎯 Applying neutral feedback fallback (score-only)")
                                neutral_results = self.system.apply_prediction_feedback(
                                    0.0, 
                                    None, 
                                    None  # CORRECTED: No actual_tokens for neutral feedback
                                )
                                print("✅ Applied neutral feedback (manual extraction failed)")
                        except Exception:
                            # CORRECTED: Final fallback (score-only)
                            print(f"🎯 Applying final neutral feedback fallback (score-only)")
                            neutral_results = self.system.apply_prediction_feedback(
                                0.0, 
                                None, 
                                None  # CORRECTED: No actual_tokens for final fallback
                            )
                            print("✅ Applied neutral feedback (all parsing failed)")

                except Exception as e:
                    print(f"❌ Error getting LLM feedback: {str(e)}")
                    # CORRECTED: Apply neutral feedback as fallback (score-only)
                    print(f"🎯 Applying error fallback neutral feedback (score-only)")
                    neutral_results = self.system.apply_prediction_feedback(
                        0.0, 
                        None, 
                        None  # CORRECTED: No actual_tokens for error fallback
                    )
                    print("✅ Applied neutral feedback (error fallback)")

                # Step 6: Update curriculum progress (PRESERVED)
                if curriculum_topic and curriculum_topic != "general knowledge":
                    print(f"\nUpdating progress for topic: {curriculum_topic}")
                    self.topic_progress[curriculum_topic] = self.topic_progress.get(curriculum_topic, 0.0) + 0.05

                    if self.topic_progress[curriculum_topic] >= self.acceptable_progress_level:
                        self.covered_topics.add(curriculum_topic)
                        session_stats["topics_covered"] += 1
                        print(f"\n🎓 Topic '{curriculum_topic}' marked as covered!")
                    else:
                        remaining = self.acceptable_progress_level - self.topic_progress[curriculum_topic]
                        print(f"📊 Topic progress: {self.topic_progress[curriculum_topic]:.2f}/{self.acceptable_progress_level} ({remaining:.2f} more to cover)")

                # ENHANCED: Display session stats with feedback impact
                elapsed_time = time.time() - session_stats["start_time"]
                print(f"\n--- Enhanced Session Stats ---")
                print(f"🎯 Topics covered: {session_stats['topics_covered']}")
                print(f"💬 Interactions: {session_stats['interactions']}")
                print(f"👍 Positive feedback: {session_stats['positive_feedback']}")
                print(f"👎 Negative feedback: {session_stats['negative_feedback']}")

                # ADDED: Feedback effectiveness metrics
                if session_stats["interactions"] > 0:
                    feedback_ratio = session_stats["positive_feedback"] / max(1, session_stats["interactions"])
                    avg_impact = session_stats["feedback_impact_total"] / session_stats["interactions"]
                    change_rate = session_stats["predictions_changed"] / session_stats["interactions"]

                    print(f"📈 Positive feedback ratio: {feedback_ratio:.2f}")
                    print(f"💪 Average feedback impact: {avg_impact:.3f}")
                    print(f"🔄 Prediction change rate: {change_rate:.2f}")
                    print(f"🎉 Total prediction changes: {session_stats['predictions_changed']}")

                print(f"⏰ Session time: {elapsed_time:.1f} seconds")
                print(f"🧠 Learning mode: {'Beam Search' if use_beam_search else 'Simple'}")
                print(f"🔧 Feedback system: WORKING (immediate impact)")
                print(f"🎯 Token extraction: CONSISTENT (always available)")
                print(f"-------------------")

                # Periodic save (PRESERVED)
                if session_stats["interactions"] % 10 == 0:
                    print("💾 Performing periodic save...")
                    try:
                        if hasattr(self.system, 'save'):
                            self.system.save()
                        print("✅ Save completed")
                    except Exception as e:
                        print(f"Warning - Error in periodic save: {str(e)}")

                print("\n--- Iteration completed successfully ---")

        except KeyboardInterrupt:
            print("\n⚠️ Session interrupted. Saving progress...")
            try:
                if hasattr(self.system, 'save'):
                    self.system.save()
                print("✅ Progress saved successfully.")
            except Exception as e:
                print(f"❌ Error saving progress: {str(e)}")
        except Exception as e:
            print(f"\n❌ Unexpected error in training session: {str(e)}")
            print("Attempting to save progress...")
            try:
                if hasattr(self.system, 'save'):
                    self.system.save()
                print("✅ Progress saved despite error.")
            except Exception as e2:
                print(f"❌ Error saving progress: {str(e2)}")

    # ===================================================================
# UPDATE 2: Enhanced Interactive Learning (main.py)  
# ===================================================================

    def _handle_interactive_learning(self):
        """
        WORKING: Interactive learning with immediate visible feedback.

        FIXES APPLIED:
        1. SIMPLIFIED feedback processing
        2. IMMEDIATE impact display
        3. CLEAR before/after comparison
        4. WORKING prediction changes
        """
        print("\n🎓 WORKING INTERACTIVE LEARNING MODE")
        print("Features:")
        print("• IMMEDIATE feedback impact (no database issues)")
        print("• STRONG corrections (50% positive boost, 40% negative reduction)")
        print("• VISIBLE prediction changes")
        print("• DIRECT node targeting")
        print("\nCommands:")
        print("• Type text for prediction")
        print("• 'beam on/off' - toggle beam search")
        print("• 'stats' - show feedback statistics")
        print("• 'exit' - return to menu")

        use_beam_search = False

        try:
            while True:
                print(f"\n🔍 Mode: {'Beam Search' if use_beam_search else 'Simple'} | Working Feedback: ACTIVE")
                print("🗣️  Enter text or command:")
                user_input = input().strip()

                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'beam on':
                    use_beam_search = True
                    print("✅ Beam search enabled")
                    continue
                elif user_input.lower() == 'beam off':
                    use_beam_search = False
                    print("✅ Simple prediction enabled")
                    continue
                elif user_input.lower() == 'stats':
                    # Show feedback statistics
                    stats = self.system.get_feedback_stats()
                    self._display_feedback_stats(stats)
                    continue
                
                if not user_input:
                    print("Please enter some text.")
                    continue
                
                try:
                    # STEP 1: Process input for context
                    context_embedding = self.system.trie_memory.context_window.current_context_embedding
                    result = self.system.process_input(user_input, 0.5)
                    print(f"✅ Processed: {len(result['tokens'])} tokens")

                    # STEP 2: Generate prediction WITH tracking
                    query_sequence_embedding = result.get('query_sequence_embedding', None)
                    continuation, confidence = self.system.predict_continuation(
                        user_input, context_embedding, query_sequence_embedding, use_beam_search=use_beam_search
                    )

                    if continuation:
                        output_tokens = []
                        for i, token in enumerate(continuation):
                            if token in {'?', '!', '.', ',', ';', ':'} and output_tokens:
                                output_tokens[-1] += token  # attach to previous token
                            else:
                                output_tokens.append(token)

                        prediction_text = ' '.join(output_tokens)
                        method = "BEAM" if use_beam_search else "SIMPLE"
                        print(f"\n🔮 {method} PREDICTION:")
                        print(f"   Input: '{user_input}'")
                        print(f"   Output: '{prediction_text}'")
                        print(f"   Confidence: {confidence:.3f}")

                        # STEP 3: Get feedback (SIMPLIFIED)
                        print(f"\n📝 FEEDBACK:")
                        print(f"   👍 Good prediction? Enter positive number (0.1 to 1.0)")
                        print(f"   👎 Bad prediction? Enter negative number (-1.0 to -0.1)")
                        print(f"   📝 Want to correct? Enter the correct response")
                        print(f"   ⏭️  Skip? Press Enter")

                        feedback_input = input("Feedback: ").strip()

                        if feedback_input:
                            # Parse feedback
                            try:
                                # Try parsing as number
                                feedback_score = float(feedback_input)
                                feedback_score = max(-1.0, min(1.0, feedback_score))
                                user_correction = None
                                print(f"📊 Feedback score: {feedback_score:.3f}")

                            except ValueError:
                                # Not a number, treat as correction
                                feedback_score = -0.8  # Strong negative for correction
                                user_correction = feedback_input
                                print(f"📝 Correction provided: '{user_correction}'")
                                print(f"📊 Auto feedback score: {feedback_score:.3f}")

                            # STEP 4: Apply feedback (WORKING)
                            print(f"\n🔧 Applying immediate feedback...")
                            actual_tokens = self.system._tokenize(user_correction) if user_correction else None
                            feedback_results = self.system.apply_prediction_feedback(
                                feedback_score, user_correction, actual_tokens=actual_tokens
                            )

                            # STEP 5: Display CLEAR results
                            if 'error' not in feedback_results:
                                print(f"✅ FEEDBACK APPLIED SUCCESSFULLY:")
                                print(f"   🎯 Type: {feedback_results['correction_type'].upper()}")
                                print(f"   📊 Nodes affected: {feedback_results['nodes_affected']}")
                                print(f"   💪 Immediate impact strength: {feedback_results['immediate_impact_strength']:.3f}")

                                # Show specific changes
                                if feedback_results.get('confidence_changes'):
                                    print(f"   🎯 CONFIDENCE CHANGES:")
                                    for change in feedback_results['confidence_changes'][:3]:  # Show first 3
                                        print(f"      '{change['token']}': {change['before']:.3f} → {change['after']:.3f} ({change['change']:+.3f})")

                                if feedback_results.get('activation_changes'):
                                    print(f"   ⚡ ACTIVATION CHANGES:")
                                    for change in feedback_results['activation_changes'][:3]:  # Show first 3
                                        print(f"      '{change['token']}': {change['before']:.3f} → {change['after']:.3f} ({change['change']:+.3f})")

                                if feedback_results.get('correction_learned'):
                                    print(f"   📚 Correction learned and reinforced")

                            else:
                                print(f"❌ Feedback error: {feedback_results['error']}")

                            # STEP 6: Test impact by making same prediction again
                            print(f"\n🔄 Testing feedback impact...")
                            test_continuation, test_confidence = self.system.predict_continuation(
                                user_input, context_embedding, query_sequence_embedding, use_beam_search=use_beam_search
                            )

                            if test_continuation:
                                output_tokens = []
                                for i, token in enumerate(test_continuation):
                                    if token in {'?', '!', '.', ',', ';', ':'} and output_tokens:
                                        output_tokens[-1] += token  # attach to previous token
                                    else:
                                        output_tokens.append(token)

                        
                                test_prediction = ' '.join(output_tokens)
                                print(f"🔮 NEW PREDICTION AFTER FEEDBACK:")
                                print(f"   Input: '{user_input}'")
                                print(f"   Output: '{test_prediction}'")
                                print(f"   Confidence: {test_confidence:.3f}")

                                # Compare predictions
                                if test_prediction != prediction_text:
                                    print(f"🎉 SUCCESS: Prediction CHANGED due to feedback!")
                                elif abs(test_confidence - confidence) > 0.1:
                                    print(f"📈 PARTIAL: Confidence changed by {test_confidence - confidence:+.3f}")
                                else:
                                    print(f"⚠️  Limited visible change (may need stronger feedback)")

                            self.session_stats['interactions'] += 1
                        else:
                            print("⏭️  Feedback skipped")

                    else:
                        print("🤔 No prediction available - continue training!")

                except Exception as e:
                    logger.error(f"Error in working interactive learning: {e}")
                    print(f"❌ Error: {e}")

        except KeyboardInterrupt:
            print("\n⚠️ Working interactive learning interrupted")

    def _display_feedback_stats(self, stats: Dict[str, Any]):
        """Display clear feedback statistics."""
        print("\n📊 FEEDBACK SYSTEM STATISTICS")
        print("=" * 40)

        try:
            if 'error' in stats:
                print(f"❌ Error: {stats['error']}")
                return

            if 'message' in stats:
                print(f"ℹ️  {stats['message']}")
                return

            total = stats.get('total_feedback', 0)
            positive = stats.get('positive_feedback', 0)
            negative = stats.get('negative_feedback', 0)

            print(f"📈 FEEDBACK SUMMARY:")
            print(f"  • Total feedback: {total}")
            print(f"  • Positive feedback: {positive}")
            print(f"  • Negative feedback: {negative}")

            if total > 0:
                print(f"  • Positive ratio: {stats.get('positive_ratio', 0.0):.1%}")
                print(f"  • Negative ratio: {stats.get('negative_ratio', 0.0):.1%}")

            print(f"\n🎯 IMPACT SUMMARY:")
            print(f"  • Nodes strengthened: {stats.get('nodes_strengthened', 0)}")
            print(f"  • Nodes weakened: {stats.get('nodes_weakened', 0)}")
            print(f"  • Average impact: {stats.get('average_impact_per_feedback', 0.0):.3f}")
            print(f"  • System responsiveness: {stats.get('system_responsiveness', 'UNKNOWN')}")

            print(f"\n🔧 SYSTEM STATUS:")
            tracked = stats.get('last_prediction_tracked', False)
            print(f"  • Last prediction tracked: {'✅ YES' if tracked else '❌ NO'}")

        except Exception as e:
            logger.error(f"Error displaying feedback stats: {e}")
            print(f"❌ Error displaying stats: {e}")


    def _show_system_insights(self):
        """PRESERVED: Display comprehensive system insights"""
        print("\n🧠 SYSTEM INSIGHTS")
        print("=" * 50)
        
        try:
            insights = self.system.get_system_insights()
            
            print("📊 TRIE STATISTICS:")
            trie_stats = insights.get('trie_statistics', {})
            print(f"  • Total sequences: {trie_stats.get('total_sequences', 0)}")
            print(f"  • Context window size: {trie_stats.get('context_window_size', 0)}")
            print(f"  • Current context available: {trie_stats.get('current_context_available', False)}")
            
            print("\n⚡ HIGH-ACTIVATION NODES:")
            high_activation = insights.get('high_performing_nodes', {}).get('high_activation', [])
            if high_activation:
                for i, node in enumerate(high_activation[:5], 1):
                    print(f"  {i}. '{node['token']}' - Activation: {node['activation_level']:.3f}, "
                          f"Access count: {node['access_count']}")
            else:
                print("  No high-activation nodes found")
            
            print("\n🏆 HIGH-REWARD NODES:")
            high_reward = insights.get('high_performing_nodes', {}).get('high_reward', [])
            if high_reward:
                for i, node in enumerate(high_reward[:5], 1):
                    print(f"  {i}. '{node['token']}' - Avg reward: {node['avg_reward']:.3f}, "
                          f"Total reward: {node['total_reward']:.3f}, Frequency: {node['frequency']}")
            else:
                print("  No high-reward nodes found")
            
            print("\n🔍 BEAM SEARCH CONFIG:")
            beam_config = insights.get('beam_search_config', {})
            print(f"  • Beam width: {beam_config.get('beam_width', 'N/A')}")
            print(f"  • Max generation length: {beam_config.get('max_generation_length', 'N/A')}")
            
            scoring_weights = beam_config.get('scoring_weights', {})
            if scoring_weights:
                print("  • Scoring weights:")
                for component, weight in scoring_weights.items():
                    print(f"    - {component.title()}: {weight:.2f}")
            
            identity = insights.get('identity_context', {})
            if identity:
                print(f"\n👤 IDENTITY CONTEXT:")
                for key, value in identity.items():
                    print(f"  • {key}: {value}")
            
            activation_context = insights.get('activation_context', {})
            if activation_context:
                print(f"\n⚡ ACTIVATION CONTEXT:")
                print(f"  • Status: {'ENABLED' if activation_context.get('enabled') else 'DISABLED'}")
                print(f"  • Description: {activation_context.get('description', 'N/A')}")
            
            logger.info("Displayed comprehensive system insights")
            
        except Exception as e:
            logger.error(f"Error showing system insights: {e}")
            print(f"❌ Error retrieving insights: {e}")

    def _handle_hf_training(self):
        """PRESERVED: Handle training on HuggingFace datasets"""
        if not self.hf_integration:
            print("❌ HuggingFace integration not available.")
            return

        try:
            print("\n📚 HUGGINGFACE DATASET TRAINING (Progressive Prediction Mode)")
            print("Available datasets:")
            print("1. PersonaChat (conversational)")
            print("2. Daily Dialog (daily conversations)")
            print("3. WikiText-2 (encyclopedia text)")
            print("4. Reddit Dataset (reddit posts and comments)")
            print("5. Train on all datasets")

            choice = input("Select dataset (1-5): ").strip()

            max_samples = input("Max samples per dataset (default 1000): ").strip()
            max_samples = int(max_samples) if max_samples else 1000

            if choice in ["1", "2", "3", "4"]:
                if choice == "1":
                    count = self.hf_integration.process_dataset(
                        "bavard/personachat_truecased", None, "persona_chat", 
                        max_samples, predictor_instance=self, shuffle=True
                    )
                    print(f"✅ Processed PersonaChat dataset: {count} samples")
                elif choice == "2":
                    count = self.hf_integration.process_dataset(
                        "daily_dialog", None, "daily_dialog", 
                        max_samples, predictor_instance=self, shuffle=True
                    )
                    print(f"✅ Processed Daily Dialog dataset: {count} samples")
                elif choice == "3":
                    count = self.hf_integration.process_dataset(
                        "wikitext", "wikitext-2-raw-v1", "wikitext_2_raw", 
                        max_samples, predictor_instance=self, shuffle=True
                    )
                    print(f"✅ Processed WikiText-2 dataset: {count} samples")
                elif choice == "4":
                    count = self.hf_integration.process_dataset(
                        "SocialGrep/the-reddit-dataset-dataset", "comments", "comments", 
                        max_samples, predictor_instance=self, shuffle=True
                    )
                    print(f"✅ Processed Reddit dataset: {count} samples")
                
                if count:
                    self.session_stats['dataset_samples'] += count

            elif choice == "5":
                results = {}
                for dataset_name, config, friendly_name in self.hf_integration.dataset_configs:
                    try:
                        count = self.hf_integration.process_dataset(
                            dataset_name, config, friendly_name,
                            max_samples, predictor_instance=self, shuffle=True
                        )
                        results[friendly_name] = count
                        if count:
                            self.session_stats['dataset_samples'] += count
                    except Exception as e:
                        logger.error(f"Error processing {friendly_name}: {e}")
                        results[friendly_name] = 0

                print("\n✅ Training Results:")
                for dataset, count in results.items():
                    print(f"   {dataset}: {count if count is not None else 0} samples")
            else:
                print("❌ Invalid choice.")

        except Exception as e:
            logger.error(f"Error in HuggingFace training: {e}")
            print(f"❌ Training error: {e}")

    def _handle_statistics(self):
        """PRESERVED: Display comprehensive learning statistics"""
        print("\n📊 LEARNING STATISTICS")
        print("=" * 60)
        
        try:
            runtime = time.time() - self.session_stats['start_time']
            print("🕒 SESSION STATISTICS:")
            print(f"  • Runtime: {runtime:.1f} seconds")
            print(f"  • Interactions: {self.session_stats['interactions']}")
            print(f"  • Predictions Made: {self.session_stats['predictions']}")
            print(f"  • Dataset Samples: {self.session_stats['dataset_samples']}")
            
            insights = self.system.get_system_insights()
            
            print(f"\n💾 MEMORY STATISTICS:")
            trie_stats = insights.get('trie_statistics', {})
            print(f"  • Total sequences stored: {trie_stats.get('total_sequences', 0)}")
            print(f"  • Context window size: {trie_stats.get('context_window_size', 0)}")
            print(f"  • Context available: {trie_stats.get('current_context_available', False)}")
            
            print(f"\n⚡ ACTIVATION ANALYSIS:")
            high_activation = insights.get('high_performing_nodes', {}).get('high_activation', [])
            if high_activation:
                print(f"  • Top {len(high_activation)} most active nodes:")
                for i, node in enumerate(high_activation, 1):
                    print(f"    {i}. '{node['token']}' - Activation: {node['activation_level']:.3f}")
                
                avg_activation = sum(node['activation_level'] for node in high_activation) / len(high_activation)
                print(f"  • Average activation level: {avg_activation:.3f}")
            else:
                print("  • No high-activation nodes detected")
            
            print(f"\n🏆 REWARD ANALYSIS:")
            high_reward = insights.get('high_performing_nodes', {}).get('high_reward', [])
            if high_reward:
                print(f"  • Top {len(high_reward)} highest-reward nodes:")
                for i, node in enumerate(high_reward, 1):
                    print(f"    {i}. '{node['token']}' - Avg: {node['avg_reward']:.3f}, "
                          f"Total: {node['total_reward']:.3f}")
                
                avg_reward = sum(node['avg_reward'] for node in high_reward) / len(high_reward)
                total_reward = sum(node['total_reward'] for node in high_reward)
                print(f"  • Average reward: {avg_reward:.3f}")
                print(f"  • Total accumulated reward: {total_reward:.3f}")
            else:
                print("  • No high-reward nodes detected")
            
            print(f"\n🔍 PREDICTION SYSTEM:")
            beam_config = insights.get('beam_search_config', {})
            print(f"  • Beam width: {beam_config.get('beam_width', 'N/A')}")
            print(f"  • Max generation length: {beam_config.get('max_generation_length', 'N/A')}")
            
            scoring_weights = beam_config.get('scoring_weights', {})
            if scoring_weights:
                print("  • Scoring component weights:")
                for component, weight in scoring_weights.items():
                    print(f"    - {component.replace('_', ' ').title()}: {weight:.2f}")
            
            if self.session_stats['interactions'] > 0:
                print(f"\n📈 LEARNING EFFICIENCY:")
                predictions_per_interaction = self.session_stats['predictions'] / self.session_stats['interactions']
                print(f"  • Predictions per interaction: {predictions_per_interaction:.2f}")
                
                if runtime > 0:
                    interactions_per_minute = (self.session_stats['interactions'] / runtime) * 60
                    print(f"  • Interactions per minute: {interactions_per_minute:.1f}")
            
            identity = insights.get('identity_context', {})
            if identity:
                print(f"\n👤 IDENTITY CONTEXT:")
                for key, value in identity.items():
                    print(f"  • {key}: {value}")
            
            logger.info("Displayed comprehensive learning statistics")
            
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            print(f"❌ Error retrieving statistics: {e}")

    def _handle_save_load(self):
        """PRESERVED: Handle save/load model functionality"""
        print("\n💾 SAVE/LOAD MODEL")
        print("1. Save model")
        print("2. Load model")
        
        choice = input("Select option (1-2): ").strip()
        
        if choice == "1":
            try:
                print("✅ Model automatically saved to LMDB database")
                logger.info("Model save completed (LMDB handles persistence)")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                print(f"❌ Save error: {e}")
                
        elif choice == "2":
            try:
                print("✅ Model loads automatically from LMDB database on startup")
                logger.info("Model load completed (LMDB loads automatically)")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                print(f"❌ Load error: {e}")
        else:
            print("❌ Invalid choice")


# Add this to test feedback correction database
def test_feedback_database(system):
    """Test if feedback corrections database is working."""
    try:
        diagnosis = system.feedback_system.diagnose_corrections_database()
        print("\n🔍 FEEDBACK DATABASE DIAGNOSIS:")
        print(f"   Database initialized: {diagnosis.get('database_initialized', False)}")
        print(f"   Database accessible: {diagnosis.get('database_accessible', False)}")
        print(f"   Total records: {diagnosis.get('total_records', 0)}")
        print(f"   Record types: {diagnosis.get('record_types', {})}")
        if diagnosis.get('sample_keys'):
            print(f"   Sample keys: {diagnosis['sample_keys']}")
        if 'error' in diagnosis:
            print(f"   ❌ Error: {diagnosis['error']}")
    except Exception as e:
        print(f"❌ Database test failed: {e}")





# Example usage and testing
if __name__ == "__main__":
    logger.info("Starting Enhanced Trie-Based Predictive System")
    
    try:
        # Call this after system initialization:
        test_feedback_database(system=PredictiveSystem())
        print("🎯 Choose startup mode:")
        print("1. Interactive Menu System")
        print("2. Demo Mode (quick test)")
        
        mode = input("Enter choice (1-2): ").strip()
        
        if mode == "1":
            # Interactive menu system
            interface = UserInteraction()
            interface.run()
            
        elif mode == "2":
            # Demo mode - preserved original testing logic
            system = PredictiveSystem()
            
            training_data = [
                ("hello how are you", 0.8),
                ("hello there friend", 0.7),
                ("how are you doing", 0.9),
                ("i am doing well", 0.8),
                ("thank you very much", 0.9)
            ]
            
            print("Training the system...")
            for text, reward in training_data:
                context_embedding = system.trie_memory.context_window.current_context_embedding
                result = system.process_input(text, reward)
                print(f"Processed: '{text}' -> ID: {result['sequence_id'][:8]}...")
            
            print("\nTesting predictions...")
            test_queries = ["hello", "how are", "i am"]
            
            for query in test_queries:
                query_sequence_embedding = result.get('query_sequence_embedding', None)
                continuation, confidence = system.predict_continuation(query, context_embedding, query_sequence_embedding)
                if continuation:
                    full_prediction = f"{''.join(continuation)}"
                    print(f"Query: '{query}' -> Prediction: '{full_prediction}' (confidence: {confidence:.3f})")
                else:
                    print(f"Query: '{query}' -> No prediction found")
            
            system.close()
            logger.info("Demo completed successfully")
        else:
            print("❌ Invalid choice")
        
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        raise