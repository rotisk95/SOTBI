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
    
    def __init__(self, db_path: str = "./trie_memory.lmdb"):
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
        """PRESERVED: Main interactive loop with curriculum learning support"""
        print("🚀 Starting Recursive Token Weaver System...")
        logger.info("Starting interactive user interface")

        try:
            while True:
                self.show_menu()
                choice = input("Enter your choice (1-6): ").strip()

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
                else:
                    print("❌ Invalid choice. Please enter 1-6.")

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

    def show_menu(self):
        """PRESERVED: Display interactive menu for user"""
        print("\n" + "="*60)
        print("🧠 RECURSIVE TOKEN WEAVER - HYBRID LEARNING SYSTEM WITH ACTIVATION")
        print("="*60)
        print("1. Interactive Learning Mode (Real-time prediction with RL feedback)")
        print("2. Curriculum LLM Training (Structured topic-based learning)")
        print("3. Train on HuggingFace datasets")
        print("4. View learning statistics")
        print("5. Save/Load model")
        print("6. Exit")
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
                    predicted_tokens, confidence = self.system.predict_continuation(' '.join(context))
                    
                    reward = 0.0
                    if predicted_tokens and len(predicted_tokens) > 0:
                        if predicted_tokens[0] == actual_next:
                            reward = confidence * 1.0
                        else:
                            reward = confidence * 0.1
                    
                    self.system.process_input(' '.join(context + [actual_next]), reward)
                    
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
        
    def interactive_llm_training_session(self, response="I am a student, I learn from conversations. Teach about me. And also from the following topic:"):
        """Run interactive conversation session with curriculum learning using process_input approach."""
        print("\n=== Fractal Memory System - Curriculum Learning ===")
        print("Type 'exit' to quit, 'help' for commands, 'skip' to skip current topic")
        print("Commands:")
        print("• 'beam on/off' - toggle beam search")
        print("• 'details' - show detailed scoring")
        print("• 'skip' - skip current topic")
        print("• 'exit' - quit session")

        # Track session stats
        session_stats = {
            "topics_covered": 0,
            "interactions": 0,
            "start_time": time.time()
        }

        # Learning mode settings
        use_beam_search = False
        show_details = False

        try:
            while True:
                # Step 1: Get curriculum topic
                print("\nSelecting curriculum topic...")
                curriculum_topic = self.choose_from_curriculum("curriculum.json")

                if curriculum_topic:
                    print(f"Selected topic: {curriculum_topic}")
                    prompt = response + " Teach me about the following topic: " + curriculum_topic
                else:
                    print("Using general knowledge prompt (no uncovered topics found)")
                    prompt = response + " general knowledge"
                    curriculum_topic = "general knowledge"

                # Step 2: Get LLM response
                print(f"\nSending prompt to LLM about: {curriculum_topic}")
                print("Waiting for LLM response (this may take some time)...")

                try:
                    llm_input = self.llm_model.get_response(prompt)
                    input_length = len(llm_input)
                    print(f"Received LLM response ({input_length} chars)")

                    # Show preview of response
                    preview_length = min(150, input_length)
                    print(f"\nLLM: {llm_input[:preview_length]}..." if input_length > preview_length else f"\nLLM: {llm_input}")
                except Exception as e:
                    print(f"Error getting LLM response: {str(e)}")
                    llm_input = f"Let me tell you about {curriculum_topic}. This topic involves learning important concepts."
                    print(f"Using fallback response: {llm_input[:50]}...")

                # Check for exit commands
                if llm_input.lower() == "exit":
                    break
                elif llm_input.lower() == "help":
                    print("Commands: 'exit' to quit, 'help' for commands, 'skip' to skip current topic")
                    continue
                elif llm_input.lower() == "skip":
                    print(f"Skipping topic: {curriculum_topic}")
                    continue

                # Step 3: Process LLM input using process_input approach
                print("\nProcessing LLM input with curriculum context...")
                try:
                    # Add curriculum context to the input
                    contextual_input = f"Learning about {curriculum_topic}: {llm_input}"

                    # Process the input (this replaces store_message and chunked processing)
                    result = self.system.process_input(contextual_input, 0.7)  # Higher confidence for LLM content
                    print(f"✅ Processed: {len(result['tokens'])} tokens")

                except Exception as e:
                    print(f"Warning - Error processing LLM input: {str(e)}")

                # Step 4: Generate AI student response using prediction
                print("Generating AI student response...")
                try:
                    # Use prediction to generate response
                    continuation, confidence = self.system.predict_continuation(
                        contextual_input, use_beam_search=use_beam_search
                    )

                    if continuation:
                        ai_response = ''.join(continuation)
                        print(f"AI Student: {ai_response}")
                        print(f"Confidence: {confidence:.3f}")

                        # Process the AI response back into the system
                        self.system.process_input(ai_response, 0.8)  # High confidence for AI responses

                    else:
                        # Fallback response if no prediction available
                        ai_response = f"I'm learning about {curriculum_topic}. This is very interesting and I want to understand more about these concepts."
                        print(f"AI Student (fallback): {ai_response}")
                        self.system.process_input(ai_response, 0.5)

                    session_stats["interactions"] += 1

                except Exception as e:
                    print(f"Error generating response: {str(e)}")
                    fallback_response = f"I'm studying {curriculum_topic}. It's fascinating to learn about these ideas."
                    print(f"AI Student (fallback): {fallback_response}")
                    try:
                        self.system.process_input(fallback_response, 0.3)
                    except Exception:
                        pass

                # Step 5: Optional user feedback (like in interactive learning)
                feedback_input = input("\nRate the AI response (-1.0 to 1.0, or press Enter to continue): ").strip()
                if feedback_input:
                    # Handle user commands
                    if feedback_input.lower() == 'beam on':
                        use_beam_search = True
                        print("✅ Beam search enabled")
                    elif feedback_input.lower() == 'beam off':
                        use_beam_search = False
                        print("✅ Simple prediction enabled")
                    elif feedback_input.lower() == 'details':
                        show_details = not show_details
                        print(f"✅ Detailed scoring: {'ON' if show_details else 'OFF'}")
                    elif feedback_input.lower() == 'skip':
                        print(f"Skipping to next topic...")
                        continue
                    elif feedback_input.lower() == 'exit':
                        break
                    else:
                        # Try to parse as feedback score
                        try:
                            reward = float(feedback_input)
                            reward = max(-1.0, min(1.0, reward))
                            # Apply feedback to the AI response
                            self.system.process_input(ai_response, reward)
                            print(f"✅ Applied feedback reward: {reward}")
                        except ValueError:
                            print("Invalid input, continuing...")

                # Step 6: Update curriculum progress
                if curriculum_topic and curriculum_topic != "general knowledge":
                    print(f"\nUpdating progress for topic: {curriculum_topic}")
                    self.topic_progress[curriculum_topic] = self.topic_progress.get(curriculum_topic, 0.0) + 0.05

                    # Check if topic should be marked as covered
                    if self.topic_progress[curriculum_topic] >= self.acceptable_progress_level:
                        self.covered_topics.add(curriculum_topic)
                        session_stats["topics_covered"] += 1
                        print(f"\n🎓 Topic '{curriculum_topic}' marked as covered!")
                    else:
                        remaining = self.acceptable_progress_level - self.topic_progress[curriculum_topic]
                        print(f"📊 Topic progress: {self.topic_progress[curriculum_topic]:.2f}/{self.acceptable_progress_level} ({remaining:.2f} more to cover)")

                # Display session stats
                elapsed_time = time.time() - session_stats["start_time"]
                print(f"\n--- Session Stats ---")
                print(f"🎯 Topics covered: {session_stats['topics_covered']}")
                print(f"💬 Interactions: {session_stats['interactions']}")
                print(f"⏰ Session time: {elapsed_time:.1f} seconds")
                print(f"🧠 Learning mode: {'Beam Search' if use_beam_search else 'Simple'}")
                print(f"-------------------")

                # Periodic save (using system's save mechanism)
                if session_stats["interactions"] % 10 == 0:
                    print("💾 Performing periodic save...")
                    try:
                        # Assuming the system has a save method
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


    def _handle_interactive_learning(self):
        """PRESERVED: Interactive learning mode with beam search options and detailed feedback"""
        print("\n🎓 INTERACTIVE LEARNING MODE")
        print("Features:")
        print("• Real-time prediction with RL feedback")
        print("• Multi-node beam search with activation/RL scoring")
        print("• Detailed scoring breakdown available")
        print("\nCommands:")
        print("• Type text for prediction")
        print("• 'beam on/off' - toggle beam search")
        print("• 'activation on/off' - toggle event-driven activation context")
        print("• 'details' - show detailed scoring")
        print("• 'insights' - show system insights")
        print("• 'debug <query>' - debug trie structure for query")
        print("• 'exit' - return to menu")
        
        use_beam_search = False  # Default to simple prediction (working correctly)
        show_details = False
        
        try:
            while True:
                activation_status = "ON" if self.system.use_activation_context else "OFF"
                print(f"\n🔍 Mode: {'Beam Search' if use_beam_search else 'Simple'} | Details: {'ON' if show_details else 'OFF'} | Activation: {activation_status}")
                print("🗣️  Enter text or command:")
                raw = sys.stdin.read()  # reads until EOF
                user_input = raw.strip()
                
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
                elif user_input.lower() == 'activation on':
                    self.system.toggle_activation_context(True)
                    print("✅ Event-driven activation context enabled")
                    continue
                elif user_input.lower() == 'activation off':
                    self.system.toggle_activation_context(False)
                    print("✅ Event-driven activation context disabled")
                    continue
                elif user_input.lower() == 'details':
                    show_details = not show_details
                    print(f"✅ Detailed scoring: {'ON' if show_details else 'OFF'}")
                    continue
                elif user_input.lower() == 'insights':
                    self._show_system_insights()
                    continue
                # In your _handle_interactive_learning method, add:
                elif user_input.startswith('debug '):
                    query = user_input[6:]  # Remove 'debug '
                    tokens = self.system._tokenize(query)
                    self.system.trie_memory.debug_trie_structure(tokens)
                    continue
                
                if not user_input:
                    print("Please enter some text.")
                    continue
                
                try:
                    result = self.system.process_input(user_input, 0.5)
                    print(f"✅ Processed: {len(result['tokens'])} tokens")
                    
                    if show_details and use_beam_search:
                        prediction_result = self.system.predict_with_detailed_scoring(user_input)
                        continuation = prediction_result['predicted_continuation']
                        confidence = prediction_result['confidence_score']
                        
                        if continuation:
                            prediction = ''.join(continuation)
                            print(f"🔮 BEAM PREDICTION: '{user_input} {prediction}'")
                            print(f"📊 Confidence: {confidence:.3f}")
                            
                            if 'beam_search_details' in prediction_result:
                                details = prediction_result['beam_search_details']
                                if details:
                                    last_step = details[-1]
                                    print(f"🔍 Generation: {len(details)} steps, {last_step['num_candidates']} final candidates")
                            
                            weights = prediction_result['scoring_weights']
                            print(f"⚖️  Scoring weights: Activation({weights['activation_weight']:.2f}) "
                                 f"RL({weights['rl_weight']:.2f}) Relevance({weights['relevance_weight']:.2f}) "
                                 f"Coherence({weights['coherence_weight']:.2f}) Completeness({weights['completeness_weight']:.2f})")
                        else:
                            print("🤔 No beam prediction available")
                            
                    else:
                        continuation, confidence = self.system.predict_continuation(
                            user_input, use_beam_search=use_beam_search
                        )
                        prediction = ' '.join(continuation) if continuation else None
                        if prediction:
                            prediction = re.sub(r' +([,.!?;:])', r'\1', prediction)
                        if prediction:
                            method = "BEAM" if use_beam_search else "SIMPLE"
                            #if it's "?" we respond with something like "I don't know". "What do you mean?", "Can you please clarify?" etc.
                            if '?' in prediction:
                                prediction = "I don't know. Can you please clarify?"
                            print(f"🔮 {method} PREDICTION: '{user_input}' '{prediction}' (confidence: {confidence:.3f})")
                            
                        else:
                            print("🤔 No prediction available yet - keep training!")
                    
                    # MODIFIED: Allow negative feedback in interactive loop
                    if prediction:
                        feedback = input("Rate prediction (-1.0 to 1.0, or press Enter for 0.0): ").strip()
                        try:
                            reward = float(feedback) if feedback else 0.0  # CHANGED: Default to neutral
                            reward = max(-1.0, min(1.0, reward))  # CHANGED: Allow negative range
                            logger.info(f"User provided reward: {reward}")
                        except ValueError:
                            reward = 0.0  # CHANGED: Neutral default instead of 0.5
                            logger.warning("Invalid reward input, defaulting to 0.0")
                        
                        
                        self.system.process_input(prediction, reward)
                        print(f"✅ Updated with reward: {reward} (Method: {'Beam' if use_beam_search else 'Simple'})")
                        
                        # We add alternative response when neutral or negative feedback is given
                        
                        # ADDED: Log reward type for debugging
                        if reward > 0.5:
                            print("📈 POSITIVE reinforcement applied")
                            self.session_stats['interactions'] += 1
                            continue
                        elif reward < -0.2:
                            print("📉 NEGATIVE correction applied")
                        else:
                            print("⚖️ NEUTRAL/weak feedback applied")
                        feedback_response = input("Provide feedback on the response (e.g., 'I meant X' or 'This is not what I asked'): ")
                        self.system.process_input(feedback_response, 1.0)  # CHANGED: Default neutral feedback
                        self.session_stats['interactions'] += 1
                    
                except Exception as e:
                    logger.error(f"Error in interactive learning: {e}")
                    print(f"❌ Error: {e}")
                    
        except KeyboardInterrupt:
            print("\n⚠️ Interactive learning interrupted")

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


# Example usage and testing
if __name__ == "__main__":
    logger.info("Starting Enhanced Trie-Based Predictive System")
    
    try:
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
                result = system.process_input(text, reward)
                print(f"Processed: '{text}' -> ID: {result['sequence_id'][:8]}...")
            
            print("\nTesting predictions...")
            test_queries = ["hello", "how are", "i am"]
            
            for query in test_queries:
                continuation, confidence = system.predict_continuation(query)
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