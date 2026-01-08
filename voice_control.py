import sys
import inspect
from poses import RaisedArmPose, WarriorIIPose

import speech_recognition as sr


import difflib

import speech_recognition as sr
import difflib

class VoiceCommandCenter:
    def __init__(self, pose_map, threshold=0.5):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.pose_map = pose_map
        self.active_pose_instance = RaisedArmPose()
        self.threshold = threshold
        self.stop_listening = False
        
        # Adjust for noise once at startup to save time in the loop
        with self.microphone as source:
            print("Adjusting for background noise... please stay quiet.")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Adjustment complete.")

    def find_best_pose(self, spoken_text):
        pose_names = list(self.pose_map.keys())
        matches = difflib.get_close_matches(spoken_text, pose_names, n=1, cutoff=self.threshold)
        return matches[0] if matches else None

    def listen_loop(self):
        print("Voice Command Center Active. Say 'Hey Coach'...")
        while not self.stop_listening:
            try:
                with self.microphone as source:
                    # Short timeout so we don't wait forever on silence
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=4)
                
                # Recognition block
                text = self.recognizer.recognize_google(audio).lower()
                print(f"DEBUG: Heard '{text}'") # Log everything heard for debugging

                if "hey coach" in text:
                    command = text.split("hey coach")[-1].strip()
                    print(f"DEBUG: Extracted command '{command}'")
                    
                    best_match = self.find_best_pose(command)
                    if best_match:
                        print(f"LOG: Switching to {best_match}")
                        self.active_pose_instance = self.pose_map[best_match]()
                    else:
                        print(f"LOG: Confident match not found for '{command}'")
            
            except sr.WaitTimeoutError:
                # This happens if no audio is detected within the 2s timeout
                continue 
            except sr.UnknownValueError:
                # This happens if the audio was unintelligible
                continue
            except sr.RequestError as e:
                print(f"LOG: API Error; {e}")
                time.sleep(1) # Wait a bit before retrying if internet is down
            except Exception as e:
                print(f"LOG: Unexpected error: {e}")
                continue