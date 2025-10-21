import os
import sys      
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import mimetypes
import rembg
import io
import shutil
import cv2
import numpy as np
import re
import json
sample_name = ''
# --- (setup function is the same) ---
def setup():
    """Load API key and configure the generative AI model."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("🔴 ERROR: GOOGLE_API_KEY not found.")
        print("Please create a .env file and add GOOGLE_API_KEY=YOUR_KEY_HERE")
        return False
    try:
        genai.configure(api_key=api_key)
        print("✅ Google API Key configured successfully.")
        return True
    except Exception as e:
        print(f"🔴 ERROR configuring Google AI: {e}")
        return False

# --- (deconstruct_pdf function is the same) ---
def deconstruct_pdf(pdf_path):
    """Opens a PDF, extracts text, saves images, and returns a list of assets."""
    print(f"\n--- Starting Deconstruction of {pdf_path} ---")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"🔴 ERROR: Could not open file {pdf_path}.")
        print(f"Details: {e}")
        return None, None

    output_dir = "extracted_assets"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Assets will be saved to: {output_dir}/")

    print("\n--- Extracting All Text ---")
    full_text = ""
    for page_num, page in enumerate(doc):
        full_text += page.get_text()
    print(f"Total characters found: {len(full_text)}")

    print("\n--- Extracting and Saving All Images ---")
    image_assets = []
    
    for page_num, page in enumerate(doc):
        image_list = page.get_images(full=True)
        if image_list:
            print(f"Found {len(image_list)} images on page {page_num + 1}")
            for img_index, img in enumerate(image_list):
                xref = img[0] 
                bbox = page.get_image_bbox(img)
                
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    if image_ext == "png":
                        try:
                            test_img = Image.open(io.BytesIO(image_bytes))
                            if test_img.format == "JPEG":
                                image_ext = "jpeg"
                        except Exception:
                            pass 

                    image_filename = f"page{page_num + 1}_img{img_index + 1}_xref{xref}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    image_assets.append({
                        "page_num": page_num + 1,
                        "img_index": img_index + 1,
                        "xref": xref,
                        "bbox": bbox,
                        "local_path": image_path,
                        "category": None
                    })
                    print(f"  > Saved: {image_path}")
                    
                except Exception as e:
                    print(f"    ... 🔴 ERROR extracting image {xref}: {e}")

    print(f"\nFound and saved {len(image_assets)} total images.")
    print("--- Deconstruction Complete ---")
    
    doc.close()
    return full_text, image_assets

# --- (classify_images function is the same) ---
def classify_images(image_assets):
    """
    Uses Gemini to classify a list of image assets.
    """
    print("\n--- Starting AI Image Classification ---")
    
    model = genai.GenerativeModel('gemini-2.5-flash-lite') 
    
    classification_prompt = """
    You are a PDF processing specialist for the apparel industry.
    Analyze this image from a clothing tech pack and classify it into ONE of the following categories:
    
    - product_photo: (A clean photo of the garment itself, front, back, or side)
    - construction_diagram: (A technical drawing, sketch, or photo with annotations, arrows, or callouts for assembly)
    - measurement_table: (A grid or table of measurements, sizes, and POMs)
    - logo: (A company or brand logo)
    
    Respond with ONLY the category name (e.g., "product_photo").
    """
    
    for asset in image_assets:
        print(f"Classifying: {asset['local_path']}...")
        
        try:
            img = Image.open(asset['local_path'])
            response = model.generate_content([classification_prompt, img])
            category = response.text.strip().lower()
            asset['category'] = category
            print(f"  > Result: {category}")
            
        except Exception as e:
            print(f"  > 🔴 ERROR classifying image: {e}")
            asset['category'] = 'classification_failed'
            
    print("--- AI Classification Complete ---")
    return image_assets


# --- (pixel_based_crop function is the same) ---
def pixel_based_crop(image_path):
    """
    Surgically crops the table by finding all grid line pixels
    and ignoring the header region.
    """
    try:
        print(f"    ... Starting PIXEL-BASED crop for {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print("    ... 🔴 Crop failed: Could not read image.")
            return

        img_height, img_width = img.shape[:2]
        header_threshold_y = img_height * 0.25
        print(f"    ... Ignoring header region (top {header_threshold_y:.0f} pixels)")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 15, -2)

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        
        grid_mask = h_lines + v_lines
        y_coords, x_coords = np.where(grid_mask > 0)
        
        if len(y_coords) == 0:
            print("    ... 🔴 Crop failed: No grid lines found.")
            return

        valid_pixels = []
        for y, x in zip(y_coords, x_coords):
            if y > header_threshold_y:
                valid_pixels.append((y, x))
        
        if not valid_pixels:
            print(f"    ... 🔴 Crop failed: All grid lines were in the header. No table found.")
            return
        
        valid_y = [p[0] for p in valid_pixels]
        valid_x = [p[1] for p in valid_pixels]
        
        x_min = min(valid_x)
        y_min = min(valid_y)
        x_max = max(valid_x)
        y_max = max(valid_y)
        
        padding = 5
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_width, x_max + padding)
        y_max = min(img_height, y_max + padding)

        print(f"    ... New pixel-based crop box: [({x_min},{y_min}), ({x_max},{y_max})]")
        
        cropped_img = img[y_min:y_max, x_min:x_max]
        cv2.imwrite(image_path, cropped_img)
        print(f"    ... Pixel-based crop successful. File overwritten.")

    except Exception as e:
        print(f"    ... 🔴 Pixel-based crop failed: {e}")

# --- (process_images function is the same) ---
def process_images(classified_assets, doc_path):
    """
    Processes classified images based on their category.
    """
    print("\n--- Starting Specialized Image Processing ---")
    
    processed_dir = "processed_assets"
    os.makedirs(processed_dir, exist_ok=True)
    print(f"Processed assets will be saved to: {processed_dir}/")
    
    doc = fitz.open(doc_path)
    
    final_asset_list = []
    
    for asset in classified_assets:
        category = asset['category']
        
        if category == 'logo' or category == 'classification_failed':
            print(f"Skipping: {asset['local_path']} (Category: {category})")
            continue

        new_filename = f"{category}_page{asset['page_num']}_img{asset['img_index']}.png"
        processed_path = os.path.join(processed_dir, new_filename)
        asset['processed_path'] = processed_path
        # ★★★ Use placeholder URL for final JSON ★★★
        asset['public_url'] = f"https://example.com/placeholder/{new_filename}" 
        
        print(f"Processing: {asset['local_path']} as {category}...")
        
        try:
            if category == 'product_photo':
                with open(asset['local_path'], 'rb') as i:
                    input_bytes = i.read()
                    output_bytes = rembg.remove(input_bytes)
                    with open(processed_path, 'wb') as o:
                        o.write(output_bytes)
                    print(f"  > Saved background-free image to: {processed_path}")
            
            elif category == 'construction_diagram':
                page = doc[asset['page_num'] - 1]
                bbox = asset['bbox']
                pix = page.get_pixmap(clip=bbox, dpi=300)
                pix.save(processed_path)
                print(f"  > Saved pixel-perfect crop to: {processed_path}")

            elif category == 'measurement_table':
                page = doc[asset['page_num'] - 1]
                bbox = asset['bbox']
                pix = page.get_pixmap(clip=bbox, dpi=300)
                pix.save(processed_path)
                print(f"  > Saved initial crop to: {processed_path}")
                
                pixel_based_crop(processed_path)

            final_asset_list.append(asset)

        except Exception as e:
            print(f"  > 🔴 ERROR processing {asset['local_path']}: {e}")

    doc.close()
    print("--- Specialized Processing Complete ---")
    return final_asset_list


# --- (extract_table_data function is the same) ---
def extract_table_data(table_image_path):
    """
    Takes the path to the cropped table image and uses Gemini
    to extract the structured data. Returns (sizes_list, measurements_list)
    """
    print(f"\n--- Extracting Data from Table using Gemini: {table_image_path} ---")
    
    try:
        img = Image.open(table_image_path)
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        ocr_prompt = """
        You are a data extraction specialist. Analyze this image of a measurement table from a clothing tech pack.
        Extract ALL data from it and return it as a single, valid JSON object.
        
        The JSON must have two keys: "sizes" and "measurements".
        
        1. "sizes": This must be an array of strings containing all the size names from the table's columns (e.g., "30", "31", "32", "34", "36").
        2. "measurements": This must be an array of objects. Each object represents one row (one Point of Measure).
        
        Each measurement object must have:
        - "pom": The name of the measurement (e.g., "WAISTBAND RELAXED", "1/2 LEG OPENING").
        - "code": The "POM" or "CODE" number/letter (e.g., "A", "B", "001"). If not present, generate one ("001", "002").
        - "tol_plus": The tolerance+ value. If not present, default to 0.5. MUST be a number.
        - "tol_minus": The tolerance- value. If not present, default to 0.5. MUST be a number.
        - "values": An array of numbers (NOT strings) corresponding to the sizes. Convert all fractions to decimals (e.g., "15 1/2" -> 15.5).
        
        CRITICAL RULES:
        - The number of "values" in each measurement must exactly match the number of "sizes".
        - Ensure all measurement values are numbers (float or int), not strings.
        - Do not include the "GRADE" or "TOL" columns in the main "sizes" array.
        
        Respond with ONLY the valid JSON object and nothing else.
        """
        
        response = model.generate_content([ocr_prompt, img])
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        print("--- RAW GEMINI OCR OUTPUT ---")
        print(json_text)
        print("--- END RAW GEMINI OCR OUTPUT ---")
        
        data = json.loads(json_text)
        
        sizes = data.get("sizes", [])
        measurements = data.get("measurements", [])
        
        if not sizes or not measurements:
            raise ValueError("Gemini response was missing 'sizes' or 'measurements' key.")

        # --- Data Validation & Cleaning ---
        cleaned_measurements = []
        for i, meas in enumerate(measurements):
            # Ensure code exists
            if "code" not in meas or not meas["code"]:
                meas["code"] = f"{i+1:03d}"
            # Ensure tolerances are numbers
            meas["tol_plus"] = float(meas.get("tol_plus", 0.5))
            meas["tol_minus"] = float(meas.get("tol_minus", 0.5))
            # Ensure values are numbers and match size count
            if len(meas.get("values", [])) != len(sizes):
                print(f"⚠️ Warning: Mismatch between number of sizes ({len(sizes)}) and values ({len(meas.get('values', []))}) for POM '{meas.get('pom', 'Unknown')}'. Skipping POM.")
                continue
            
            numeric_values = []
            valid_row = True
            for val in meas["values"]:
                 try:
                    numeric_values.append(float(val))
                 except (ValueError, TypeError):
                    print(f"⚠️ Warning: Could not convert measurement value '{val}' to number for POM '{meas.get('pom', 'Unknown')}'. Skipping POM.")
                    valid_row = False
                    break
            if valid_row:
                 meas["values"] = numeric_values
                 cleaned_measurements.append(meas)

        print(f"  > Successfully parsed {len(sizes)} sizes and {len(cleaned_measurements)} measurements.")
        
        return sizes, cleaned_measurements # Return only sizes and measurements

    except Exception as e:
        print(f"🔴 ERROR during Gemini OCR: {e}")
        return None, None


# --- ★★★ UPDATED extract_text_data FUNCTION (EXTRACTS SIZE CLASS) ★★★ ---
def extract_text_data(full_text):
    """
    Uses a targeted Gemini call to extract unstructured data like product info,
    colorways, and the descriptive size class name from the full PDF text.
    Returns (product_info_dict, colorways_list, size_class_string)
    """
    print("\n--- Extracting Data from Full Text using Gemini ---")
    
    # Default values in case of error
    default_product_info = {"style_number": "---", "product_name": "---", "brand": "---"}
    default_colorways = [{"name": "---", "number": "---", "pantone": "---", "hex": "000000"}]
    default_size_class = "Default Size Class"

    try:
        model = genai.GenerativeModel('gemini-2.5-flash-lite') 
        
        extraction_prompt = f"""
        Analyze the following text extracted from a clothing tech pack PDF.
        Extract the required information and return it as a single, valid JSON object.
        
        The JSON must have three keys: "product_info", "colorways", and "size_class".
        
        1. "product_info": An object containing:
           - "style_number": The style number or product code (e.g., "MSP26B26"). Default to "---" if not found.
           - "product_name": The product name or description (e.g., "MONTAUK SHORT"). Default to "---" if not found.
           - "brand": The brand name, if mentioned. Default to "---" if not found.
           
        2. "colorways": An array of objects. Each object represents one color found.
           Each colorway object must have:
           - "name": The primary color name (e.g., "GROVE"). Default to "---" if not found.
           - "number": The color number or code (e.g., "220"). Default to "---" if not found.
           - "pantone": The Pantone code (e.g., "19-4038 TCX"), if available. Default to "---" if not found.
           - "hex": Generate a standard 6-digit hex code (no #) based on the color name (e.g., "GROVE" -> "228B22"). If no name, use "000000".

        3. "size_class": A descriptive string representing the size category found in the text, 
           often near the measurement table (e.g., "Men's Numerical - 32\" Inseam", "WOMENS ALPHA S-XL"). 
           Default to "Default Size Class" if not clearly specified.
           
        CRITICAL RULES:
        - Search the entire text for clues for all fields.
        - If multiple colors are mentioned, create multiple colorway objects.
        - Ensure hex code is always a 6-digit string without '#'.
        
        Respond with ONLY the valid JSON object and nothing else.
        
        --- TEXT TO ANALYZE ---
        {full_text}
        """
        
        response = model.generate_content(extraction_prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        print("--- RAW GEMINI TEXT EXTRACTION OUTPUT ---")
        print(json_text)
        print("--- END RAW GEMINI TEXT EXTRACTION OUTPUT ---")
        
        data = json.loads(json_text)
        
        product_info = data.get("product_info", default_product_info)
        colorways = data.get("colorways", [])
        size_class = data.get("size_class", default_size_class)
        
        if not colorways:
             colorways = default_colorways

        color_map = {"grove": "228B22", "navy": "000080", "black": "000000", "white": "FFFFFF", "grey": "808080", "heather grey": "C0C0C0"}
        for color in colorways:
            if "hex" not in color or not color["hex"]:
                 color_name_lower = color.get("name", "---").lower()
                 color["hex"] = color_map.get(color_name_lower, "000000")

        print(f"  > Successfully parsed product info, {len(colorways)} colorways, and size class '{size_class}'.")

        return product_info, colorways, size_class # Return all three

    except Exception as e:
        print(f"🔴 ERROR during Gemini Text Extraction: {e}")
        # Return default structure on error
        return default_product_info, default_colorways, default_size_class


# --- ★★★ UPDATED build_final_json FUNCTION (MATCHES PROMPT 2 EXACTLY) ★★★ ---
def build_final_json(product_info, colorways, all_assets, table_sizes, table_measurements, size_class_name, full_text):
    """
    Assembles all the extracted pieces into the final JSON structure
    matching the schema from Prompt 2 EXACTLY.
    """
    print("\n--- Assembling Final JSON ---")
    
    final_data = {}
    
    # --- Basic Info & Placeholders ---
    final_data["document_id"] = "PLACEHOLDER_DOC_ID" # Should come from webhook
    final_data["beproduct_package_url"] = "PLACEHOLDER_ZIP_URL" # Needs generation step

    # --- Product Info ---
    final_data["product_information"] = {
        "style_number": product_info.get("style_number", "---"),
        "product_name": product_info.get("product_name", "---"),
        "brand": product_info.get("brand", "---")
        # "[DYNAMIC FIELDS FROM WEBHOOK JSON DEFINITIONS]": "---" # Needs webhook context
    }

    # --- Images ---
    # Use placeholder URLs instead of local paths
    final_data["front_image"] = next((a['public_url'] for a in all_assets if a['category'] == 'product_photo'), None)
    final_data["back_image"] = None # Placeholder
    final_data["side_image"] = None # Placeholder
    
    construction_urls = [a['public_url'] for a in all_assets if a['category'] == 'construction_diagram']
    final_data["construction"] = {
        "details": "---", # Extract this in extract_text_data if needed
        "images": construction_urls
    }
    
    product_image_urls = [a['public_url'] for a in all_assets if a['category'] == 'product_photo']
    final_data["product_images"] = product_image_urls

    # --- Colorways ---
    formatted_colorways = []
    for cw in colorways:
        color_number = cw.get("number", "---")
        color_name = cw.get("name", "---")
        pantone = cw.get("pantone", "---")
        hex_code = cw.get("hex", "000000")
        
        marketing_name = f"{color_number} - {color_name}" if color_number != "---" and color_name != "---" else "---"
        
        formatted_colorways.append({
            "id": "", "imageHeaderId": None, "unlinkImage": False,
            "fields": [
                {"id": "colorNumber", "value": color_number},
                {"id": "colorName", "value": color_name},
                {"id": "primary", "value": hex_code}, # Ensure 6-digit hex without #
                {"id": "brand_marketing_name", "value": marketing_name},
                {"id": "pantone", "value": pantone},
                {"id": "secondary", "value": ""}, # Placeholder
                {"id": "color_number", "value": color_number}, 
                {"id": "color_name", "value": color_name}     
            ]
        })
    final_data["colorways"] = formatted_colorways

    # --- Size Classes & Measurements ---
    if table_sizes and table_measurements:
        simple_sizes = table_sizes
        measurements = table_measurements
        
        # Build sizeClasses section exactly as per Prompt 2
        size_objects = []
        for i, size_name in enumerate(simple_sizes):
            is_sample = (i == len(simple_sizes) // 2) 
            
            # Find measurements for *this specific size* for the 'fields' part
            # This part of Prompt 2 schema seems redundant if 'measurements' section exists,
            # but we will follow it.
            size_fields = {}
            for meas in measurements:
                 pom_name = meas.get("pom", f"pom_{i}")
                 if len(meas.get("values", [])) > i:
                    # Prompt 2 asks for string value here, oddly. Let's provide it.
                    size_fields[pom_name] = str(meas["values"][i]) 
                 else:
                    size_fields[pom_name] = "N/A"

            size_objects.append({
                "name": size_name, # Simple size name (e.g., "30")
                "price": 0, "currency": "USD", "unitOfMeasure": "PCS",
                "comments": "", # Extract if available
                "isSampleSize": is_sample, "sizeIndex": i,
                "hideSize": False, 
                "fields": size_fields # Per Prompt 2 structure
            })
            
        final_data["sizeClasses"] = [{
            "id": "",
            "sizeClass": size_class_name, # Descriptive name from text extraction
            "active": True, "isDefault": True,
            "sizes": size_objects
        }]
        
        # Build measurements (POM) section exactly as per Prompt 2
        poms = []
        for meas in measurements:
            pom_name = meas.get("pom", "---")
            pom_code = meas.get("code", f"{len(poms)+1:03d}") 
            # Use .get with default for safety
            tol_minus = float(meas.get("tol_minus", 0.5)) 
            tol_plus = float(meas.get("tol_plus", 0.5))
            values = meas.get("values", [])
            
            graded_spec = []
            # CRITICAL: Match Prompt 2 exactly - numeric values here
            if len(values) == len(simple_sizes):
                 for size_name, value in zip(simple_sizes, values):
                    try:
                        numeric_value = float(value)
                    except (ValueError, TypeError):
                        numeric_value = 0.0 
                        print(f"⚠️ Final JSON Warning: Could not convert measurement value '{value}' to number for POM '{pom_name}', size '{size_name}'. Using 0.0.")
                        
                    graded_spec.append({"size": size_name, "value": numeric_value}) # Numeric value
            else:
                 # This warning was handled in extract_table_data, no need to repeat
                 continue 

            poms.append({
                "id": "", "blockPomRowId": None, "linkedWithBlockPom": False,
                "code": pom_code, 
                "pointOfMeasure": pom_name, # Exact name from OCR
                "tolMinus": tol_minus, 
                "tolPlus": tol_plus,
                "fields": [], # Always empty per Prompt 2
                "gradedSpec": graded_spec
            })
            
        final_data["measurements"] = {
            "sizeClass": size_class_name, # MUST match sizeClasses.sizeClass EXACTLY
            "poms": poms
        }
    else:
        # Handle case where no table data was extracted
        final_data["sizeClasses"] = []
        final_data["measurements"] = {"sizeClass": size_class_name, "poms": []} # Use extracted name if available

    # --- All Extracted Data ---
    final_data["all_extracted_data"] = {
        "raw_full_text_length": len(full_text) 
    }
    
    # --- Final Output ---
    # Ensure strict adherence to Prompt 2's order might require manual ordering if needed
    final_json_str = json.dumps(final_data, indent=2)
    print("--- FINAL JSON STRUCTURE ---")
    print(final_json_str)
    print("--- END FINAL JSON STRUCTURE ---")
    return final_json_str

# --- ★★★ UPDATED MAIN FUNCTION WITH JSON FILE OUTPUT ★★★ ---
def main():
    log_file_path = "process_log.txt"
    json_output_path = "final_output.json" # New output file
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    print(f"--- Starting Process ---")
    
    final_json_result = None # Variable to hold the final JSON string
    
    try:
        if setup():
            pdf_file = sample_name
            
            full_text, image_assets = deconstruct_pdf(pdf_file) 
            
            if not image_assets:
                print("🚨 No images found in PDF. Exiting.")
                return

            classified_assets = classify_images(image_assets)
            final_assets = process_images(classified_assets, pdf_file)

            table_image_asset = next((a for a in final_assets if a['category'] == 'measurement_table'), None)
            table_sizes = None
            table_measurements = None
            if table_image_asset:
                table_sizes, table_measurements = extract_table_data(table_image_asset['processed_path'])
            else:
                print("\n⚠️ No measurement table found or processed.")
                
            # Extract text data AND size class name
            product_info, colorways, size_class_name = extract_text_data(full_text) 
            
            # Build the final JSON using all extracted parts
            final_json_result = build_final_json(
                product_info, 
                colorways, 
                final_assets, 
                table_sizes, 
                table_measurements, 
                size_class_name, 
                full_text
            ) 

        else:
            print("🚨 Please fix the setup issues above.")
            
    except Exception as e:
        print(f"🔥🔥🔥 UNHANDLED ERROR: {e} 🔥🔥🔥")
        import traceback
        traceback.print_exc(file=sys.stderr) 
        
    finally:
        print(f"--- Process Complete. Log saved to {log_file_path} ---")
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        
        # ★★★ Write ONLY the final JSON to its own file ★★★
        if final_json_result:
            try:
                with open(json_output_path, 'w') as f_json:
                    # Re-parse and dump for pretty printing, ensures valid JSON written
                    json_data = json.loads(final_json_result) 
                    json.dump(json_data, f_json, indent=2)
                print(f"\n✅ Process Complete. Full log in '{log_file_path}'. Final JSON output in '{json_output_path}'.")
            except Exception as e:
                 print(f"\n🔥 Error writing final JSON file: {e}")
        else:
             print(f"\n⚠️ Process finished but no final JSON was generated (check '{log_file_path}' for errors).")


if __name__ == "__main__":
    sample_name = "sample3.pdf"
    main()

