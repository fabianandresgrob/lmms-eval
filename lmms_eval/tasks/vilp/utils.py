"""Utility functions for ViLP (Probing Visual Language Priors in VLMs) benchmark."""

from typing import Any

import numpy as np


def vilp_doc_to_visual(doc: dict[str, Any]) -> list:
    """Extract the appropriate image based on _image_idx.

    Args:
        doc: Document with _image_idx set by ViLPTask expansion

    Returns:
        List containing the appropriate RGB image
    """
    image_idx = doc.get("_image_idx", 1)
    image_key = f"image{image_idx}"

    if image_key in doc and doc[image_key] is not None:
        return [doc[image_key].convert("RGB")]
    return []


def vilp_doc_to_target(doc: dict[str, Any]) -> str:
    """Extract the appropriate answer based on _image_idx.

    Args:
        doc: Document with _image_idx set by ViLPTask expansion

    Returns:
        Target answer string
    """
    image_idx = doc.get("_image_idx", 1)
    answer_key = f"answer{image_idx}"
    return str(doc.get(answer_key, ""))


def normalize_output(output: str) -> str:
    """Normalize output for comparison.

    Converts written numbers to digits, maps synonyms, and converts
    plurals to singular forms.

    Args:
        output: The output string to normalize

    Returns:
        Normalized output string
    """
    # Map written-out numbers to digit strings
    number_mapping = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    # Map known synonyms to a single canonical term
    synonym_mapping = {
        "refrigerator": "fridge",
        "refrigerators": "fridge",
        "stove": "oven",
        "alligator": "crocodile",
        "porpoise": "dolphin",
        "automobile": "car",
        "nyc": "new york city",
        "la": "los angeles",
        "usa": "united states",
        "co2": "carbon dioxide",
        "o2": "oxygen",
        "n2": "nitrogen",
        "h2o": "water",
        "tortoise": "turtle",
        "motorbike": "motorcycle",
        "cellphone": "phone",
        "telephone": "phone",
        "pc": "computer",
        "tv": "television",
        "tap": "faucet",
        "aeroplane": "airplane",
        "cubic": "cube",
        "cubical": "cube",
        "cubes": "cube",
        "cuboids": "cube",
        "cuboid": "cube",
        "square": "cube",
        "squares": "cube",
        "striped": "stripes",
        "checkered": "checkerboard",
        "polka-dots": "spots",
        "dalmatian": "dog",
        "triangular": "triangle",
        "circular": "round",
        "circle": "round",
        "circles": "round",
        "spherical": "round",
        "spheres": "round",
        "sphere": "round",
        "triangles": "triangle",
        "logs": "wood",
        "zigzag": "curved",
        "hexagonal": "hexagon",
        "bud": "flower",
        "hippopotamus": "hippo",
        "rhinoceros": "rhino",
        "bike": "bicycle",
        "schoolbus": "bus",
        "boat": "ship",
        "boats": "ship",
        "sailboat": "ship",
        "airship": "ship",
        "donut": "torus",
        "donuts": "torus",
        "wallaby": "kangaroo",
        "teacup": "cup",
        "teapot": "kettle",
        "rooster": "chicken",
        "roosters": "chicken",
        "raven": "crow",
        "vineyard": "vine",
        "crystal": "glass",
        "hay": "straw",
        "fireplace": "oven",
        "carbon dioxide": "carbondioxide",
        "aircondition": "AC",
        "airconditioner": "AC",
        "air-conditioner": "AC",
        "t-rex": "dinosaur",
        "trex": "dinosaur",
        "man": "person",
        "woman": "person",
        "people": "person",
        "men": "person",
        "women": "person",
        "multicolored": "rainbow",
        "thatch": "straw",
        "plane": "airplane",
        "goggles": "glasses",
        "night-vision": "glasses",
        "blossoms": "flower",
        "brush": "eraser",
        "serpent": "snake",
        "dots": "spots",
        "binoculars": "glasses",
        "slippers": "shoe",
        "slipper": "shoe",
        "pillow": "cushion",
        "hexagons": "hexagon",
        "ukulele": "guitar",
        "cello": "violin",
        "steel": "metal",
        "cucumber": "pickle",
        "galaxy": "space",
        "underwater": "sea",
        "ocean": "sea",
        "faceted": "diamond",
        "jewelry": "diamond",
        "jewelries": "diamond",
        "backpack": "bag",
        "squid": "octopus",
        "kitten": "cat",
        "octagonal": "octagon",
        "candy": "lolipop",
        "pipeline": "pipe",
        "dragonfruit": "pitaya",
        "new york": "new york city",
        "eyesight": "eye",
    }

    # Convert plural forms to singular
    plural_singular_mapping = {
        "butterflies": "butterfly",
        "bees": "bee",
        "ants": "ant",
        "wasps": "wasp",
        "kangaroos": "kangaroo",
        "koalas": "koala",
        "wombats": "wombat",
        "trees": "tree",
        "books": "book",
        "goats": "goat",
        "squirrels": "squirrel",
        "rabbits": "rabbit",
        "pandas": "panda",
        "giraffes": "giraffe",
        "lions": "lion",
        "tigers": "tiger",
        "cows": "cow",
        "horses": "horse",
        "cats": "cat",
        "dogs": "dog",
        "whales": "whale",
        "sharks": "shark",
        "dolphins": "dolphin",
        "flowers": "flower",
        "leaves": "leaf",
        "knives": "knife",
        "wolves": "wolf",
        "mice": "mouse",
        "geese": "goose",
        "children": "child",
        "teeth": "tooth",
        "feet": "foot",
        "fungi": "fungus",
        "stimuli": "stimulus",
        "media": "medium",
        "octopi": "octopus",
        "cacti": "cactus",
        "diamonds": "diamond",
        "bricks": "brick",
        "flame": "fire",
        "winds": "wind",
        "wheels": "wheel",
        "chickens": "chicken",
        "fireflies": "firefly",
        "beaks": "beak",
        "needles": "needle",
        "spinners": "spinner",
        "clouds": "cloud",
        "earthquakes": "earthquake",
        "seals": "seal",
        "pencils": "pencil",
        "petals": "petal",
        "forks": "fork",
        "seahorses": "seahorse",
        "keys": "key",
        "carrots": "carrot",
        "crayons": "crayon",
        "skyscrapers": "skyscraper",
        "birds": "bird",
        "bicycles": "bicycle",
        "watches": "watch",
        "lemons": "lemon",
        "pipes": "pipe",
        "bubbles": "bubble",
        "camels": "camel",
        "stripes": "stripe",
        "lungs": "lung",
        "gills": "gill",
        "feathers": "feather",
        "scales": "scale",
        "lollipops": "lolipop",
        "lollipop": "lolipop",
        "lolipops": "lolipop",
        "drums": "drum",
        "ropes": "rope",
        "shoes": "shoe",
        "bushes": "bush",
        "elephants": "elephant",
        "porcupines": "porcupine",
        "clocks": "clock",
        "antelopes": "antelope",
        "eyes": "eye",
        "chameleons": "chameleon",
        "rockets": "rocket",
        "turbines": "turbine",
        "ostriches": "ostrich",
        "pumpkins": "pumpkin",
        "shrubs": "shrub",
        "fields": "field",
    }

    # Preprocess the output: lowercase and strip trailing whitespace/punctuation
    output = str(output).lower().strip()
    if output.endswith("."):
        output = output[:-1]  # remove trailing period if present

    # Apply the three mappings in sequence
    output = number_mapping.get(output, output)
    output = synonym_mapping.get(output, output)
    output = plural_singular_mapping.get(output, output)

    return output


def vilp_doc_to_text(
    doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, str] | None = None
) -> str:
    """Format question text.

    Args:
        doc: Document containing question field
        lmms_eval_specific_kwargs: Optional mode and prompt template

    Returns:
        Formatted question string
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    question = doc["question"]
    mode = lmms_eval_specific_kwargs.get("mode", "with_fact")

    # For "without_fact" mode, remove the first sentence (the fact)
    if mode == "without_fact" and "." in question:
        question = ".".join(question.split(".")[1:]).strip()

    prompt_template = lmms_eval_specific_kwargs.get(
        "prompt_template", "Please answer with one word: {question}"
    )

    return prompt_template.format(question=question)


def vilp_process_results(
    doc: dict[str, Any], results: list[str]
) -> dict[str, dict[str, Any]]:
    """Process model results and compute accuracy.

    Args:
        doc: Document containing ground truth answer
        results: List containing model prediction

    Returns:
        Dictionary with accuracy metrics
    """
    pred = results[0].strip().lower()
    if pred.endswith("."):
        pred = pred[:-1]

    image_idx = doc.get("_image_idx", 1)
    answer_key = f"answer{image_idx}"
    answer = str(doc[answer_key]).strip()

    # Normalize both prediction and answer
    pred_normalized = normalize_output(pred)
    answer_normalized = normalize_output(answer)

    # Check for match (excluding 'none')
    is_correct = (
        pred_normalized == answer_normalized and pred_normalized != "none"
    )

    # Return metrics for aggregation
    return {
        "vilp_score": {"image_idx": image_idx, "correct": is_correct},
        "vilp_prior": {"image_idx": image_idx, "correct": is_correct},
    }


def vilp_aggregate_score(results: list[dict[str, Any]]) -> float:
    """Aggregate ViLP score (mean accuracy on images 2 and 3).

    Args:
        results: List of result dictionaries

    Returns:
        ViLP score (mean of image2 and image3 accuracy)
    """
    # Group by question (every 3 results is one question with 3 images)
    # ViLP Score is the mean accuracy on images 2 and 3
    image2_results = []
    image3_results = []

    for result in results:
        image_idx = result["image_idx"]
        correct = result["correct"]

        if image_idx == 2:
            image2_results.append(correct)
        elif image_idx == 3:
            image3_results.append(correct)

    # Calculate mean accuracy for images 2 and 3
    all_results = image2_results + image3_results
    if len(all_results) == 0:
        return 0.0

    return float(np.mean(all_results))


def vilp_aggregate_prior(results: list[dict[str, Any]]) -> float:
    """Aggregate ViLP prior (accuracy on image 1).

    Args:
        results: List of result dictionaries

    Returns:
        ViLP prior score (image1 accuracy)
    """
    # ViLP Prior is the accuracy on image 1
    image1_results = []

    for result in results:
        image_idx = result["image_idx"]
        correct = result["correct"]

        if image_idx == 1:
            image1_results.append(correct)

    if len(image1_results) == 0:
        return 0.0

    return float(np.mean(image1_results))