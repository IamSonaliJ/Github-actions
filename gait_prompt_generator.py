import json
import os
import sys
# Make sure image_generator_functions.py is in the same dirercctory
try:
    from image_generator import load_pipeline_and_models, generate_image_and_save
except ImportError:
    print("❌ Error: Could not find image_generator_functions.py. Ensure both files are in the same folder.")
    sys.exit(1)
    
import torch
import gc

# ==========================================================
# === KEYWORD SETS (Tiered Polarity Detection) ===
# ==========================================================

EXCELLENT_KEYWORDS = {"excellent", "perfect", "unwavering", "elite", "peak", "ultimate", "mastery", "outstanding", "efficient"}
GOOD_KEYWORDS = {"solid", "steady", "reliable", "good", "balanced", "healthy", "symmetrical", "manageable","better","recovering","recovery","improved","fitness","strong","adaptable"}
MODERATE_KEYWORDS = {"moderate", "slowing", "decline", "somewhat", "slightly", "minor", "adjustment","Average"}
POOR_KEYWORDS = {"severe", "unstable", "irregular", "poor", "drastically", "major", "compromised", "fatigue", "weakened", "struggles", "Stress", "Low","Weak","Weaker"}
POSITIVE_TIER_KEYWORDS = EXCELLENT_KEYWORDS.union(GOOD_KEYWORDS)
NEGATIVE_TIER_KEYWORDS = POOR_KEYWORDS.union(MODERATE_KEYWORDS)

base_anatomy_map = {
    "CADENCE": "The (Feet:bottom surfaces and Knees: The kneecap (Patella))",
    "STEP_SYMMETRY_VARIABILITY": "The (Gastrocnemius and Soleus muscles of both legs, strictly isolated::2.0)",
    "STEP_SYMMETRY": "The (Gastrocnemius and Soleus muscles of both legs, strictly isolated::2.0)",
    "STEP_WIDTH_VARIABILITY": "hips and thighs",
    "GAIT_SCORE_DEGRADATION": "thigh and calf muscles of leg",
    "CADENCE_STABILITY": "feet and lower legs",
    "KNEE_STABILITY": "knees",
    "KNEE_FLEXION": "knees",
    "KNEE_LOAD": "knees",
    "HRV": "The Heart",
    "HR": "The Heart",
    "SPO2": "The Lungs",
    "RECOVERY_RATE":"The Heart",
    "RESPIRATORY RATE":"The Lungs"
}

visual_effect_rules = {
"excellent": "(luminous light blue and white:1.1)",
"good": "(glowing less opacity green:::1.2, maximum impact ripples:::1.2)",
"moderate": "(glowing dull yellow-orange, maximum impact ripples:::1.2)",
"poor": "(glowing subtle light-red, maximum impact ripples:::1.2)",
}


# ==========================================================
# === CORE PROMPT GENERATION FUNCTIONS ===
# ==========================================================

def get_visual_tier_by_sentiment(title: str, message: str) -> str:
    """Determines the visual tier (excellent, good, moderate, poor) based on keyword sentiment."""
    full_text = (title + " " + message).lower()
    best_positive_tier = ""
    worst_negative_tier = ""
    positive_score = 0
    negative_score = 0

    if any(kw in full_text for kw in EXCELLENT_KEYWORDS):
        positive_score += 3
        best_positive_tier = "excellent"
    elif any(kw in full_text for kw in GOOD_KEYWORDS):
        positive_score += 1
        best_positive_tier = "good"

    if any(kw in full_text for kw in POOR_KEYWORDS):
        negative_score += 3
        worst_negative_tier = "poor"
    elif any(kw in full_text for kw in MODERATE_KEYWORDS):
        negative_score += 1
        worst_negative_tier = "moderate"

    if positive_score > negative_score:
        return best_positive_tier if best_positive_tier else "good"
    elif negative_score > positive_score:
        return worst_negative_tier if worst_negative_tier else "moderate"
    else:
        return "moderate"


def generate_single_insight_prompt(insight: dict, full_data: dict) -> tuple[str, str]:
    """
    Generates a PROMPT for a SINGLE insight item.
    """
    level = insight.get("level", "").upper()
    title = insight.get("title", "").lower()
    message = insight.get("message", "").lower()

    # --- 1. DETERMINE VISUAL KEYWORDS ---
    anatomy = base_anatomy_map.get(level, None)
    visual_effect = None
    insight_desc = []
    anatomical_regions = set()

    if anatomy:
        tier = get_visual_tier_by_sentiment(title, message)
        visual_effect = visual_effect_rules.get(tier, None)

        if visual_effect:
            anatomical_regions.add(anatomy.strip().replace(',', ''))
            prompt_fragment = f" {anatomy} are {visual_effect}"
            insight_desc.append(prompt_fragment)

    # --- 2. DETECT ACTIVITY (Determine pose based on the CURRENT insight's LEVEL) ---
    summary_text = full_data.get("data", {}).get("summary", {}).get("message", "").strip().lower() # Check 'data' key first
    if not summary_text:
        summary_text = full_data.get("summary", {}).get("message", "").strip().lower() # Fallback for dummy structure
        
    context_text = summary_text + " " + " ".join([ins.get("message", "") for ins in full_data.get("insights", []) if isinstance(ins, dict)])

    if level == "SPO2":
        activity = "stand"
    elif level == "RESPIRATORY RATE":
        activity = "respiratory"
    elif level in ["HRV", "HR", "RECOVERY_RATE"]:
        activity = "standing"
    elif "run" in context_text:
        activity = "running"
    else:
        activity = "walking" # Default for all other gait metrics

    # --- 3. CONSTRUCT THE FINAL PROMPT ---
    base_style = (
        f"A Hyper-realistic 3D Anatomical illustration human body {activity}, "
        "entire body should appear as a smooth and semi-transparent, gray body texture. "
    )

    insight_text = ""
    if insight_desc:
        intro_phrase = f"only showing:"
        insight_text = f"{intro_phrase} {', '.join(insight_desc)}."

    final_prompt = base_style + insight_text + " octane render, photorealistic, smooth edges, Clean White background ."

    return final_prompt, activity


# ==========================================================
# === MAIN EXECUTION BLOCK (The Orchestrator) ===
# ==========================================================

if __name__ == "__main__":
    json_path = "sample_gait.json"
    # IMPORTANT: Update this path to where your reference images are located locally 
    # (e.g., in a subfolder called 'i_m' next to this script)
    ref_img_dir = "C:/Users/Gowtham/Downloads/Reference_images/i_m" 
    output_dir = "generated_test_outputs"
    
    # --- 0. INITIALIZE GLOBALS ---
    # Global variables are not recommended in production code, but mimic the notebook's behavior
    global pipe
    global device
    pipe = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 1. MODEL LOADING CHECK ---
    if pipe is None:
        print("Models not found in global memory. Loading models now...")
        try:
            pipe, device = load_pipeline_and_models(device=device)
            print("✅ Model loading complete. Subsequent runs will skip this step.")
        except Exception as e:
            print(f"❌ FATAL ERROR during model loading. Error: {e}")
            sys.exit(1)
    else:
        print("✅ Models already loaded in memory. Skipping load and generating images immediately.")

    # --- 2. Load the Analysis JSON Data ---
    try:
        with open(json_path, "r") as f:
            gait_data = json.load(f)
        
        # Determine the correct key for insights based on the JSON structure
        if "data" in gait_data and isinstance(gait_data["data"], dict) and "insights" in gait_data["data"]:
             insights_to_process = gait_data["data"]["insights"]
        elif "insights" in gait_data: # Fallback for dummy structure
             insights_to_process = gait_data["insights"]
        else:
             insights_to_process = []

    except Exception as e:
        print(f"❌ Error loading or parsing JSON: {e}")
        sys.exit(1)

    if not insights_to_process:
        print("❌ Error: No insights found in the JSON file. Cannot proceed with image generation.")
        sys.exit(1)

    print("=======================================================")
    print(f"🚀 Starting sequential image generation for {len(insights_to_process)} insights.")
    print("=======================================================")

    # --- 3. THE LOOP: Process Each Insight Sequentially ---
    for i, insight in enumerate(insights_to_process):
        insight_title = insight.get("title", f"Insight {i+1}")

        # A. Generate the specific prompt for this insight
        final_prompt, activity = generate_single_insight_prompt(insight, gait_data)

        print(f"\n📝 Insight {i+1} ({insight_title}): Generated Prompt ({activity} mode):")
        print(f"   {final_prompt}")

        # B. Call the image generation function
        try:
            generate_image_and_save(
                pipe=pipe,
                prompt=final_prompt,
                ref_img_dir=ref_img_dir,
                output_dir=output_dir,
                title=insight_title
            )
            
            # CRITICAL: Additional cleanup after each generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"❌ ERROR generating image for {insight_title}. Skipping. Error: {e}")
            continue

    print("\n✅ ALL INSIGHTS PROCESSED. Check the 'generated_test_outputs' directory.")
