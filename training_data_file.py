#!/usr/bin/env python3
"""
Conversation Data Processor for Training System
Handles large conversation JSON files with streaming processing and extracts training data
"""

import json
import logging
import os
import sys
import ijson
from pathlib import Path
from typing import Dict, List, Generator, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

# Configure logging for transparency
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversation_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DecimalJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle Decimal objects in training data
    
    ADDED: Fix for 'Object of type Decimal is not JSON serializable' error
    JUSTIFICATION: ChatGPT export data contains decimal.Decimal timestamps that must be serialized
    PRESERVES: All existing data structure and content - only converts Decimal to float for JSON
    """
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)  # Convert Decimal to float for JSON serialization
        return super().default(obj)

@dataclass
class ConversationMessage:
    """Structure for individual conversation messages"""
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: Optional[str] = None
    message_id: Optional[str] = None

@dataclass
class ProcessedConversation:
    """Structure for processed conversation data"""
    conversation_id: str
    messages: List[ConversationMessage]
    title: Optional[str] = None
    total_messages: int = 0

class ConversationDataProcessor:
    """
    Processes conversation data files for training system integration
    Handles large files through streaming and provides memory-efficient processing
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize processor with dataset directory path
        
        Args:
            dataset_path: Path to C:\Datasets\Conversations directory
        """
        self.dataset_path = Path(dataset_path)
        self.processed_count = 0
        self.error_count = 0
        
        logger.info(f"Initializing ConversationDataProcessor with path: {self.dataset_path}")
        
        # Validate dataset path exists
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """
        Analyze the structure of available files in the dataset directory
        
        Returns:
            Dictionary containing file analysis results
        """
        logger.info("Starting file structure analysis")
        
        analysis = {
            'files_found': {},
            'total_size_mb': 0,
            'recommendations': []
        }
        
        try:
            for file_path in self.dataset_path.iterdir():
                if file_path.is_file():
                    file_size_mb = file_path.stat().st_size / (1024 * 1024)
                    analysis['files_found'][file_path.name] = {
                        'size_mb': round(file_size_mb, 2),
                        'path': str(file_path)
                    }
                    analysis['total_size_mb'] += file_size_mb
                    
                    logger.info(f"Found file: {file_path.name} ({file_size_mb:.2f} MB)")
            
            # Generate processing recommendations based on file sizes
            large_files = [name for name, info in analysis['files_found'].items() 
                          if info['size_mb'] > 100]
            
            if large_files:
                analysis['recommendations'].append(
                    f"Large files detected ({', '.join(large_files)}). Use streaming processing."
                )
            
            if 'conversations.json' in analysis['files_found']:
                analysis['recommendations'].append(
                    "conversations.json found - primary training data source"
                )
            
            logger.info(f"File structure analysis complete. Total size: {analysis['total_size_mb']:.2f} MB")
            return analysis
            
        except Exception as e:
            logger.error(f"Error during file structure analysis: {str(e)}")
            self.error_count += 1
            raise
    
    def stream_large_json(self, file_path: Path) -> Generator[Dict, None, None]:
        """
        Stream large JSON files to avoid memory issues
        
        Args:
            file_path: Path to JSON file
            
        Yields:
            Individual JSON objects from the file
        """
        logger.info(f"Starting streaming processing of: {file_path.name}")
        
        try:
            with open(file_path, 'rb') as file:
                # Handle different JSON structures
                if file_path.name == 'conversations.json':
                    # Assume conversations.json contains an array of conversation objects
                    parser = ijson.items(file, 'item')
                else:
                    # For other JSON files, try to parse as array
                    parser = ijson.items(file, 'item')
                
                for item in parser:
                    yield item
                    
        except ijson.JSONError as e:
            logger.error(f"JSON parsing error in {file_path.name}: {str(e)}")
            self.error_count += 1
            # Try alternative parsing approach
            try:
                logger.info(f"Attempting alternative parsing for {file_path.name}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if isinstance(data, list):
                        for item in data:
                            yield item
                    else:
                        yield data
            except Exception as fallback_error:
                logger.error(f"Fallback parsing failed: {str(fallback_error)}")
                raise
        except Exception as e:
            logger.error(f"Error streaming {file_path.name}: {str(e)}")
            self.error_count += 1
            raise
    
    def extract_conversation_messages(self, conversation_data: Dict) -> Optional[ProcessedConversation]:
        """
        Extract and structure conversation messages from raw data
        
        Args:
            conversation_data: Raw conversation data dictionary
            
        Returns:
            ProcessedConversation object or None if extraction fails
        """
        try:
            # Handle different conversation data structures
            messages = []
            conversation_id = conversation_data.get('id', 'unknown')
            title = conversation_data.get('title', '')
            
            # Extract messages based on structure found in model_comparisons.json
            if 'mapping' in conversation_data:
                # ChatGPT export format with mapping structure
                mapping = conversation_data['mapping']
                
                # Sort messages by creation time if available
                sorted_messages = []
                for node_id, node_data in mapping.items():
                    if node_data and 'message' in node_data:
                        message_data = node_data['message']
                        if message_data and 'content' in message_data:
                            # Handle mixed timestamp types (Decimal, None, int, str)
                            create_time = message_data.get('create_time', 0)
                            normalized_time = self._normalize_timestamp(create_time)
                            
                            sorted_messages.append((
                                normalized_time,
                                message_data
                            ))
                
                # Sort with robust error handling for timestamp comparison
                try:
                    sorted_messages.sort(key=lambda x: x[0])
                    logger.debug(f"Successfully sorted {len(sorted_messages)} messages by timestamp")
                except TypeError as e:
                    logger.warning(f"Timestamp sorting failed for conversation {conversation_id}: {str(e)}. Using original order.")
                    # Fallback: use original order without sorting
                    sorted_messages = [(0, msg[1]) for msg in sorted_messages]
                
                for _, message_data in sorted_messages:
                    content_parts = message_data.get('content', {}).get('parts', [])
                    role = message_data.get('author', {}).get('role', 'unknown')
                    
                    if content_parts and content_parts[0]:
                        messages.append(ConversationMessage(
                            content=content_parts[0],
                            role=role,
                            timestamp=message_data.get('create_time'),
                            message_id=message_data.get('id')
                        ))
            
            elif 'messages' in conversation_data:
                # Direct messages array format
                for msg in conversation_data['messages']:
                    messages.append(ConversationMessage(
                        content=msg.get('content', ''),
                        role=msg.get('role', 'unknown'),
                        timestamp=msg.get('timestamp'),
                        message_id=msg.get('id')
                    ))
            
            if messages:
                logger.debug(f"Extracted {len(messages)} messages from conversation {conversation_id}")
                return ProcessedConversation(
                    conversation_id=conversation_id,
                    messages=messages,
                    title=title,
                    total_messages=len(messages)
                )
            else:
                logger.warning(f"No messages found in conversation {conversation_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting messages from conversation: {str(e)}")
            self.error_count += 1
            return None
    
    def _normalize_timestamp(self, timestamp) -> float:
        """
        Normalize mixed timestamp types to comparable float values
        
        Args:
            timestamp: Mixed type timestamp (Decimal, None, int, str, float)
            
        Returns:
            Float timestamp value (0.0 for None/invalid values)
            
        ADDED: Fix for comparison error between Decimal and NoneType timestamps
        """
        try:
            if timestamp is None:
                return 0.0
            elif isinstance(timestamp, (int, float)):
                return float(timestamp)
            elif hasattr(timestamp, '__float__'):  # Handles Decimal objects
                return float(timestamp)
            elif isinstance(timestamp, str):
                try:
                    return float(timestamp)
                except ValueError:
                    logger.debug(f"Could not convert string timestamp '{timestamp}' to float, using 0.0")
                    return 0.0
            else:
                logger.debug(f"Unknown timestamp type {type(timestamp)}, using 0.0")
                return 0.0
        except Exception as e:
            logger.warning(f"Error normalizing timestamp {timestamp}: {str(e)}, using 0.0")
            return 0.0
    
    def process_conversations_for_training(self, max_conversations: Optional[int] = None) -> Generator[ProcessedConversation, None, None]:
        """
        Process conversations and yield training data
        
        Args:
            max_conversations: Limit number of conversations to process (None for all)
            
        Yields:
            ProcessedConversation objects ready for training
        """
        logger.info("Starting conversation processing for training data")
        
        conversations_file = self.dataset_path / 'conversations.json'
        
        if not conversations_file.exists():
            logger.error("conversations.json not found in dataset directory")
            raise FileNotFoundError("conversations.json not found")
        
        processed_count = 0
        
        try:
            for conversation_data in self.stream_large_json(conversations_file):
                if max_conversations and processed_count >= max_conversations:
                    logger.info(f"Reached maximum conversation limit: {max_conversations}")
                    break
                
                processed_conversation = self.extract_conversation_messages(conversation_data)
                
                if processed_conversation and processed_conversation.messages:
                    processed_count += 1
                    self.processed_count += 1
                    
                    if processed_count % 100 == 0:  # Log progress every 100 conversations
                        logger.info(f"Processed {processed_count} conversations...")
                    
                    yield processed_conversation
            
            logger.info(f"Conversation processing complete. Total processed: {processed_count}")
            
        except Exception as e:
            logger.error(f"Error during conversation processing: {str(e)}")
            self.error_count += 1
            raise
    
    def format_for_training_system(self, conversation: ProcessedConversation) -> List[Dict[str, str]]:
        """
        Format conversation data for training system input
        
        Args:
            conversation: ProcessedConversation object
            
        Returns:
            List of formatted training examples
        """
        training_examples = []
        
        try:
            # Create training pairs from conversation messages
            for i in range(len(conversation.messages) - 1):
                current_msg = conversation.messages[i]
                next_msg = conversation.messages[i + 1]
                
                # Create input-output pairs for training
                if current_msg.role == 'user' and next_msg.role == 'assistant':
                    training_examples.append({
                        'input': current_msg.content,
                        'output': next_msg.content,
                        'conversation_id': conversation.conversation_id,
                        'sequence_position': i
                    })
            
            logger.debug(f"Created {len(training_examples)} training examples from conversation {conversation.conversation_id}")
            return training_examples
            
        except Exception as e:
            logger.error(f"Error formatting conversation for training: {str(e)}")
            self.error_count += 1
            return []
    
    def save_training_data(self, output_path: str, max_conversations: Optional[int] = None):
        """
        Process all conversations and save formatted training data
        
        Args:
            output_path: Path to save the training data JSON file
            max_conversations: Maximum number of conversations to process
        """
        logger.info(f"Starting training data extraction and save to: {output_path}")
        
        training_data = []
        total_examples = 0
        
        try:
            for conversation in self.process_conversations_for_training(max_conversations):
                examples = self.format_for_training_system(conversation)
                training_data.extend(examples)
                total_examples += len(examples)
                
                # Save periodically to avoid memory issues
                if len(training_data) >= 10000:  # Save every 10k examples
                    self._append_to_training_file(output_path, training_data)
                    logger.info(f"Saved batch of {len(training_data)} examples. Total so far: {total_examples}")
                    training_data = []  # Clear memory
            
            # Save remaining data
            if training_data:
                self._append_to_training_file(output_path, training_data)
            
            logger.info(f"Training data extraction complete. Total examples: {total_examples}")
            logger.info(f"Processing summary - Processed: {self.processed_count}, Errors: {self.error_count}")
            
        except Exception as e:
            logger.error(f"Error during training data save: {str(e)}")
            self.error_count += 1
            raise
    
    def _append_to_training_file(self, output_path: str, data: List[Dict]):
        """
        Append training data to file (handles large datasets by appending)
        
        Args:
            output_path: Path to output file
            data: Training data to append
            
        MODIFIED: Added DecimalJSONEncoder to handle Decimal objects in ChatGPT export data
        PRESERVES: All existing file handling logic and data structure
        """
        try:
            output_file = Path(output_path)
            
            if output_file.exists():
                # Append to existing file
                logger.debug(f"Appending {len(data)} items to existing training file: {output_path}")
                with open(output_file, 'r+', encoding='utf-8') as f:
                    f.seek(0, 2)  # Go to end of file
                    f.seek(f.tell() - 1)  # Back up one character
                    f.write(',\n')  # Add comma and newline
                    for i, item in enumerate(data):
                        # MODIFIED: Added cls=DecimalJSONEncoder to handle Decimal serialization
                        # JUSTIFICATION: ChatGPT export contains decimal.Decimal timestamps that cause JSON serialization errors
                        # PRESERVES: All existing data content and structure unchanged
                        json.dump(item, f, ensure_ascii=False, cls=DecimalJSONEncoder)
                        if i < len(data) - 1:
                            f.write(',\n')
                    f.write('\n]')  # Close the JSON array
            else:
                # Create new file
                logger.info(f"Creating new training file with {len(data)} items: {output_path}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('[\n')
                    for i, item in enumerate(data):
                        # MODIFIED: Added cls=DecimalJSONEncoder to handle Decimal serialization
                        # JUSTIFICATION: Same as above - prevents TypeError: Object of type Decimal is not JSON serializable
                        # PRESERVES: All existing data content and structure unchanged
                        json.dump(item, f, ensure_ascii=False, cls=DecimalJSONEncoder)
                        if i < len(data) - 1:
                            f.write(',\n')
                    f.write('\n]')
                    
            logger.debug(f"Successfully saved {len(data)} training examples to {output_path}")
                    
        except Exception as e:
            logger.error(f"Error appending to training file: {str(e)}")
            raise

def main():
    """
    Main execution function with example usage
    """
    try:
        # Initialize processor
        processor = ConversationDataProcessor(r"C:\Datasets\Conversations")
        
        # Analyze file structure first
        logger.info("=== STARTING FILE ANALYSIS ===")
        analysis = processor.analyze_file_structure()
        
        print("\n=== FILE ANALYSIS RESULTS ===")
        for filename, info in analysis['files_found'].items():
            print(f"{filename}: {info['size_mb']} MB")
        
        print(f"\nTotal dataset size: {analysis['total_size_mb']:.2f} MB")
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"- {rec}")
        
        # Process conversations for training (limit to 1000 for testing)
        logger.info("=== STARTING CONVERSATION PROCESSING ===")
        output_path = "training_data.json"
        
        # Process with limit for initial testing
        processor.save_training_data(output_path, max_conversations=1000)
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Training data saved to: {output_path}")
        print(f"Check conversation_processor.log for detailed logs")
        
    except Exception as e:
        logger.error(f"Main execution error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()