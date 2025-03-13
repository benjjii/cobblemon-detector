import time
import pyautogui
import pytesseract
import numpy as np
from PIL import ImageGrab, Image, ImageFont, ImageDraw
import cv2
from playsound import playsound
import os
import tkinter as tk
import threading
import winsound
from difflib import get_close_matches
from datetime import datetime
import unicodedata
import requests
import zipfile
import io
import sys

# Path to Tesseract executable - update this to your installation path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# Change single TARGET_POKEMON to a list of target Pokemon
TARGET_POKEMON = ["Chien-Pao","Latios","Articuno", "Rayquaza", "Ogerpon", "Walking Wake", "Roaring", "Zapdos", "Galarian", "Roaring Moon", "Zekrom", "Glastrier", "Druddigon", "Hawlucha"]  # Add as many Pokemon names as you want

# Sound file for alert - replace with your own sound file path
ALERT_SOUND = "alert.mp3"  # Make sure this file exists in your directory

# Region of the screen to monitor (left, top, right, bottom)
# Adjusted to capture a larger area
screen_width, screen_height = pyautogui.size()
SCREEN_REGION = (
    screen_width // 6,           # left: 1/6 of screen width
    screen_height // 8,          # top: 1/8 of screen height (moved up)
    screen_width * 5 // 6,       # right: 5/6 of screen width
    screen_height * 7 // 8       # bottom: 7/8 of screen height (moved down)
)

def load_pokemon_names():
    """Load Pokemon names from the text file"""
    # Get the correct path whether running as script or executable
    if getattr(sys, 'frozen', False):
        # Running as executable
        script_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
    pokemon_file = os.path.join(script_dir, "pokemon_names.txt")
    
    # Check if file exists, if not create it with a basic list
    if not os.path.exists(pokemon_file):
        print("Pokemon names file not found. Creating a basic list...")
        with open(pokemon_file, "w") as f:
            f.write("\n".join([
                "Bulbasaur", "Ivysaur", "Venusaur", 
                # ... add more default Pokemon if needed
                "Kyogre", "Groudon", "Rayquaza", "Latios", "Latias"
            ]))
    
    # Load the Pokemon names
    with open(pokemon_file, "r") as f:
        pokemon_names = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(pokemon_names)} Pokemon names")
    return pokemon_names

# Load the Pokemon names at startup
ALL_POKEMON = load_pokemon_names()

def capture_screen_region():
    """Capture the specified region of the screen"""
    if SCREEN_REGION:
        screenshot = ImageGrab.grab(bbox=SCREEN_REGION)
    else:
        screenshot = ImageGrab.grab()
    return np.array(screenshot)

def preprocess_image(img):
    """Apply preprocessing optimized for game text"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize image to make text larger (3x)
    height, width = gray.shape
    enlarged = cv2.resize(gray, (width*3, height*3), interpolation=cv2.INTER_CUBIC)
    
    # Apply adaptive thresholding (works better for varying backgrounds)
    binary = cv2.adaptiveThreshold(enlarged, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    # Remove noise
    denoised = cv2.medianBlur(binary, 3)
    
    return denoised

def is_similar_to_target(text, target, threshold=0.8):
    """Check if text is similar to target using fuzzy matching"""
    # Convert both to lowercase for comparison
    text = text.lower()
    target = target.lower()
    
    # Exact match
    if target in text:
        return True, target
    
    # Split text into words and check each word
    words = text.split()
    matches = get_close_matches(target, words, n=1, cutoff=threshold)
    return len(matches) > 0, matches[0] if matches else None

def download_minecraft_font():
    """Download and extract the Minecraft font for better character recognition"""
    print("Downloading Minecraft font for improved OCR...")
    
    # Create directory for fonts if it doesn't exist
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minecraft_font")
    os.makedirs(font_dir, exist_ok=True)
    
    # Path to the Minecraft font file
    font_path = os.path.join(font_dir, "minecraft.ttf")
    
    # Check if font already exists
    if os.path.exists(font_path):
        print(f"Minecraft font already exists at {font_path}")
        return font_path
    
    try:
        # URL for Minecraft font (this is a common location, might need updating)
        font_url = "https://github.com/IdreesInc/Minecraft-Font/raw/master/minecraft_font.ttf"
        
        # Download the font
        response = requests.get(font_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Save the font
        with open(font_path, 'wb') as f:
            f.write(response.content)
        
        print(f"Downloaded Minecraft font to {font_path}")
        return font_path
    
    except Exception as e:
        print(f"Error downloading Minecraft font: {e}")
        print("Using system fonts instead")
        return None

def create_minecraft_character_samples(minecraft_font_path=None):
    """Create sample images of Minecraft-style characters to improve recognition"""
    print("Creating character samples for Minecraft text...")
    
    # Create directory for samples if it doesn't exist
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minecraft_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Basic Latin characters
    basic_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-: "
    
    # Minecraft-specific Unicode characters
    # These include special symbols used in Minecraft's chat and UI
    minecraft_special = "§★☆→←↑↓⬆⬇⬅➡✦✧❤❥♠♣♥♦☀☁☂☃☄★☆☇☈☉☊☋☌☍☎☏☐☑☒☓☔☕☖☗☘☙☚☛☜☝☞☟☠☡☢☣☤☥☦☧☨☩☪☫☬☭☮☯"
    
    # Combine all character sets
    all_chars = basic_chars + minecraft_special
    
    # Fonts to use - prioritize Minecraft font if available
    fonts = []
    if minecraft_font_path and os.path.exists(minecraft_font_path):
        fonts.append(("Minecraft", minecraft_font_path, 16))  # Minecraft font is usually best at 16px
    
    # Add fallback fonts
    fonts.extend([
        ("Arial", 16),
        ("Courier New", 16),  # Monospaced font similar to Minecraft
        ("Lucida Console", 16),  # Another monospaced font
    ])
    
    # Minecraft color palette (RGB)
    colors = [
        (255, 255, 255),  # White
        (0, 0, 0),        # Black
        (87, 87, 87),     # Dark Gray
        (170, 170, 170),  # Gray
        (255, 85, 85),    # Red
        (85, 255, 85),    # Green
        (255, 255, 85),   # Yellow
        (85, 85, 255),    # Blue
    ]
    
    # Generate samples for each character in each font
    for font_info in fonts:
        try:
            # Handle both custom and system fonts
            if len(font_info) == 3:
                font_name, font_path, font_size = font_info
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except IOError:
                    print(f"Font file {font_path} not found, skipping")
                    continue
            else:
                font_name, font_size = font_info
                try:
                    font = ImageFont.truetype(font_name, font_size)
                except IOError:
                    print(f"System font {font_name} not found, using default")
                    font = ImageFont.load_default()
            
            # Create a blank image
            img = Image.new('RGB', (32, 32), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            for char in all_chars:
                try:
                    # Get character name for filename
                    char_name = unicodedata.name(char, f"U+{ord(char):04X}")
                    safe_char_name = "".join(c for c in char_name if c.isalnum() or c == '_')
                    
                    # Create samples with different color combinations
                    for bg_color in colors[:2]:  # Just use black and white backgrounds
                        for fg_color in colors:
                            if fg_color != bg_color:  # Skip if foreground and background are the same
                                # Draw character
                                draw.rectangle([0, 0, 32, 32], fill=bg_color)
                                draw.text((8, 8), char, font=font, fill=fg_color)
                                
                                # Save the image
                                color_suffix = f"_{fg_color[0]}_{fg_color[1]}_{fg_color[2]}_on_{bg_color[0]}_{bg_color[1]}_{bg_color[2]}"
                                filename = f"{samples_dir}/{font_name.replace(' ', '_')}_{safe_char_name}{color_suffix}.png"
                                img.save(filename)
                
                except Exception as e:
                    print(f"Error processing character {char}: {e}")
        
        except Exception as e:
            print(f"Error creating samples for font {font_info}: {e}")
    
    print(f"Created Minecraft character samples in {samples_dir}")
    return samples_dir

def train_tesseract_for_minecraft(samples_dir):
    """Create a custom Tesseract configuration optimized for Minecraft text"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minecraft_ocr.config")
    
    with open(config_path, "w") as f:
        f.write("""
tessedit_char_whitelist ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-: §★☆→←↑↓⬆⬇⬅➡✦✧❤❥♠♣♥♦
classify_bln_numeric_mode 0
tessedit_create_hocr 0
tessedit_pageseg_mode 6
textord_force_make_prop_words 0
tessedit_char_blacklist |
""")
    
    print(f"Created custom Minecraft OCR config at {config_path}")
    return config_path

def initialize_minecraft_ocr():
    """Initialize OCR with Minecraft-specific optimizations"""
    # Download Minecraft font
    minecraft_font_path = download_minecraft_font()
    
    # Create character samples
    samples_dir = create_minecraft_character_samples(minecraft_font_path)
    
    # Create custom Tesseract config
    config_path = train_tesseract_for_minecraft(samples_dir)
    
    return config_path

def find_pokemon_on_screen(custom_config_path=None):
    """Check if any of the target Pokemon names are visible on screen"""
    # Capture the screen
    img = capture_screen_region()
    
    all_detected_text = []  # Store all detected text
    found_pokemon = []  # Store all found Pokemon
    match_info = {}  # Store what text matched to which Pokemon
    
    # Try different preprocessing methods optimized for Minecraft
    methods = [
        lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB),  # Original image
        preprocess_image,  # Enhanced preprocessing
        # Minecraft-specific preprocessing - high contrast for pixel fonts
        lambda x: cv2.threshold(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1],
        # Inverted for light text on dark backgrounds (common in Minecraft)
        lambda x: cv2.threshold(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)[1]
    ]
    
    # Load custom config if available
    custom_config = ""
    if custom_config_path and os.path.exists(custom_config_path):
        with open(custom_config_path, "r") as f:
            custom_config = f.read()
    
    for method in methods:
        processed_img = method(img)
        
        # Try multiple OCR configurations
        configs = [
            '--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-: ',
            '--oem 3 --psm 6',  # Assume uniform text block
            '--oem 3 --psm 7'   # Treat the image as a single text line
        ]
        
        # Add custom config if available
        if custom_config:
            configs.append(custom_config)
        
        for config in configs:
            text = pytesseract.image_to_string(processed_img, config=config)
            # Split text into lines and filter out empty lines and short text
            lines = [line.strip() for line in text.split('\n') 
                    if line.strip() and len(line.strip()) >= 4]  # Minimum 4 characters
            all_detected_text.extend(lines)
    
    # Remove duplicates while preserving order
    unique_text = list(dict.fromkeys(all_detected_text))
    
    # Find closest Pokemon matches for each text line
    detected_pokemon = []
    print("\nDetected text mapped to Pokemon:")
    print("------------------------------")
    for line in unique_text:
        closest_pokemon = find_closest_pokemon(line)
        if closest_pokemon:
            # Only add if it's not already in the list (avoid duplicates)
            if closest_pokemon not in detected_pokemon:
                detected_pokemon.append(closest_pokemon)
                print(f"'{line}' → '{closest_pokemon}'")
                
                # Store the match information
                if closest_pokemon in match_info:
                    match_info[closest_pokemon].append(line)
                else:
                    match_info[closest_pokemon] = [line]
    
    # Check for any of the target Pokemon
    target_found = []
    for pokemon in TARGET_POKEMON:
        if pokemon in detected_pokemon:
            target_found.append(pokemon)
    
    return target_found if target_found else None, match_info

def take_screenshot():
    """Take a screenshot of the capture region and save it"""
    # Get the correct path whether running as script or executable
    if getattr(sys, 'frozen', False):
        # Running as executable
        save_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        save_dir = os.path.dirname(os.path.abspath(__file__))
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot = ImageGrab.grab(bbox=SCREEN_REGION)
    filename = os.path.join(save_dir, f"pokemon_found_{timestamp}.png")
    screenshot.save(filename)
    return filename

def alert_user(pokemon_name, unique_matches):
    """Alert the user when a target Pokemon is found"""
    print(f"ALERT! {pokemon_name} detected on screen!")
    
    # Take screenshot
    screenshot_file = take_screenshot()
    print(f"Screenshot saved as: {screenshot_file}")
    
    # Local alert (beep)
    winsound.Beep(1000, 700)

def show_capture_overlay():
    """Show a semi-transparent red overlay on the capture region that can be resized"""
    root = tk.Tk()
    root.attributes('-alpha', 0.3)  # Set transparency
    root.attributes('-topmost', True)  # Keep window on top
    root.configure(bg='red')
    
    # Position and size the window using SCREEN_REGION
    left, top, right, bottom = SCREEN_REGION
    
    # Allow window decorations for resizing
    # root.overrideredirect(True)  # Remove this line to keep window decorations
    
    # Position and size the window
    root.geometry(f"{right-left}x{bottom-top}+{left}+{top}")
    
    # Add a label with instructions
    label = tk.Label(root, text="Resize this window to adjust capture area\nPress ESC to close", bg='red', fg='white')
    label.pack(pady=10)
    
    # Add escape key binding to close the window
    root.bind('<Escape>', lambda e: root.destroy())
    
    # Add function to update SCREEN_REGION when window is resized
    def on_resize(event):
        global SCREEN_REGION
        # Get the new window position and size
        x = root.winfo_x()
        y = root.winfo_y()
        width = root.winfo_width()
        height = root.winfo_height()
        
        # Update the SCREEN_REGION
        SCREEN_REGION = (x, y, x + width, y + height)
        print(f"Capture region updated: {SCREEN_REGION}")
        
        # Update the label text
        label.config(text=f"Capture area: {width}x{height}\nPosition: ({x},{y})\nPress ESC to close")
    
    # Bind the resize event
    root.bind("<Configure>", on_resize)
    
    return root

def monitor_pokemon():
    """Separate function to handle the Pokemon monitoring loop"""
    found_pokemon = []
    cooldown_counter = 0
    found_matches = {}  # Store the text that matched to each Pokemon
    
    try:
        while True:
            print("\n" + "="*50)  # Separator for readability
            print(f"Scanning for: {', '.join(TARGET_POKEMON)}...")
            
            # IMPROVEMENT 1: Increase capture frequency
            all_found = []
            current_matches = {}  # Store matches for this scan
            
            # IMPROVEMENT 2: Take more captures (5 instead of 3)
            for _ in range(5):  # Try 5 quick captures
                result = find_pokemon_on_screen()
                
                # Handle the return value properly
                if isinstance(result, tuple) and len(result) == 2:
                    current_found, match_info = result
                else:
                    current_found = result
                    match_info = {}
                
                # IMPROVEMENT 3: Alert immediately when found
                if current_found:
                    # Beep immediately when any target is found
                    for pokemon in current_found:
                        # Get the matched text for this Pokemon
                        matched_text = match_info.get(pokemon, [])
                        if isinstance(matched_text, list):
                            winsound.Beep(1000, 700)  # Longer, more noticeable beep
                            print(f"IMMEDIATE ALERT! {pokemon} detected!")
                
                    all_found.extend(current_found)
                    # Update matches dictionary
                    if isinstance(match_info, dict):
                        for pokemon, matched_text in match_info.items():
                            if pokemon in current_matches:
                                if isinstance(matched_text, list):
                                    current_matches[pokemon].extend(matched_text)
                                else:
                                    current_matches[pokemon].append(str(matched_text))
                            else:
                                current_matches[pokemon] = matched_text if isinstance(matched_text, list) else [str(matched_text)]
                
                # IMPROVEMENT 4: Shorter pause between captures
                time.sleep(0.05)  # Reduced pause between captures
            
            # Process results as before
            unique_found = list(dict.fromkeys(all_found)) if all_found else []
            
            if unique_found:
                for pokemon in unique_found:
                    if pokemon not in found_pokemon:
                        unique_matches = []
                        if pokemon in current_matches:
                            matches = current_matches[pokemon]
                            if isinstance(matches, list):
                                unique_matches = list(dict.fromkeys(matches))
                            else:
                                unique_matches = [str(matches)]
                        alert_user(pokemon, unique_matches)
                        found_matches[pokemon] = unique_matches
                    else:
                        if pokemon in current_matches:
                            current = current_matches[pokemon]
                            if not isinstance(current, list):
                                current = [str(current)]
                            existing = found_matches.get(pokemon, [])
                            if not isinstance(existing, list):
                                existing = [str(existing)]
                            found_matches[pokemon] = list(dict.fromkeys(existing + current))
                found_pokemon = unique_found
                cooldown_counter = 10  # Reduced cooldown
            
            # IMPROVEMENT 5: Faster main loop
            time.sleep(0.1)  # Reduced main loop sleep time
    
    except Exception as e:
        print(f"Error in monitoring: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"Pokemon Detector started. Looking for: {', '.join(TARGET_POKEMON)}...")
    print("Showing capture region overlay (Press ESC to remove)...")
    
    # Show the overlay window
    overlay = show_capture_overlay()
    
    # Start monitoring in a separate thread to keep overlay responsive
    monitoring_thread = threading.Thread(target=monitor_pokemon, daemon=True)
    monitoring_thread.start()
    
    # Start the overlay window
    overlay.mainloop()
    
    print("Pokemon Detector stopped.")

def find_closest_pokemon(text):
    """Find the closest Pokemon name to the given text"""
    # Skip short text
    if len(text) < 3:
        return None
    
    # Increase the cutoff threshold for more strict matching
    cutoff = 0.76  # Higher threshold = fewer false positives
    
    # Try exact match first (case insensitive)
    for pokemon in ALL_POKEMON:
        if pokemon.lower() == text.lower():
            return pokemon
    
    # Try substring match (if text contains a full Pokemon name)
    for pokemon in ALL_POKEMON:
        if pokemon.lower() in text.lower():
            return pokemon
    
    # Try fuzzy matching with higher threshold
    matches = get_close_matches(text.lower(), [p.lower() for p in ALL_POKEMON], n=1, cutoff=cutoff)
    if matches:
        # Find the original case version
        index = [p.lower() for p in ALL_POKEMON].index(matches[0])
        return ALL_POKEMON[index]
    
    return None

def validate_pokemon_match(pokemon, occurrences):
    """Validate a Pokemon match based on number of occurrences"""
    # For rare/legendary Pokemon, require more occurrences to reduce false positives
    legendary_pokemon = ["Kyogre", "Groudon", "Rayquaza", "Latios", "Latias", "Phione", "Manaphy"]
    
    if pokemon in legendary_pokemon:
        return occurrences >= 2  # Require at least 2 occurrences for legendary Pokemon
    else:
        return occurrences >= 1  # Regular Pokemon only need 1 occurrence

if __name__ == "__main__":
    main()
