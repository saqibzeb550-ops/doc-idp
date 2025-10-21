import os
import json
import re
import requests
import fitz  # PyMuPDF
from pathlib import Path

# --- API CALLER ---
def parse_pdf_with_landingai(pdf_path, api_key):
    """Calls the Landing AI Parse API and returns the extraction data."""
    if not api_key:
        raise ValueError("Landing AI API key is required.")
    
    url = "https://api.va.landing.ai/v1/ade/parse"
    headers = {'Authorization': f'Bearer {api_key}'}
    
    with open(pdf_path, 'rb') as f:
        files = {'document': (Path(pdf_path).name, f, 'application/pdf')}
        data = {'model': 'dpt-2-latest'}
        
        print("🚀 Sending request to Landing.AI Parse API...")
        response = requests.post(url, files=files, data=data, headers=headers, timeout=180)
    
    if response.status_code != 200:
        raise Exception(f"Landing.AI API Error: {response.status_code} - {response.text}")
        
    print("✅ Landing.AI Parse complete!")
    return response.json()

# --- FINALIZED TRANSFORMATION LOGIC ---
def transform_extraction_data(extraction_data):
    """
    A high-fidelity Python port of the original Node.js transformation script,
    with bug fixes and improved data cleaning to match the target JSON quality.
    """
    print("🔄 Transforming extraction data (Final Corrected Version)...")
    chunks = extraction_data.get('chunks', [])
    if not chunks:
        return {}

    # --- Helper Functions ---
    
    def clean_html(raw_html):
        """Removes ONLY standard HTML tags, preserving special syntax."""
        if not raw_html: return ""
        # This regex targets standard <tag> and </tag> formats
        clean = re.sub(r'</?[a-zA-Z0-9]+[^>]*>', '', raw_html)
        clean = clean.replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<')
        return clean.strip()

    def extract_title_info(chunks):
        """Refined to extract full product name correctly."""
        info = {'style_number': None, 'product_name': None}
        title_chunk = next((c for c in chunks if c.get('type') == 'text' and c.get('grounding', {}).get('page') == 0), None)
        if not title_chunk: return info

        title_text = clean_html(title_chunk.get('markdown', ''))
        
        style_patterns = [r'(MSP\w+|LSP\w+|MFA\w+)', r'([A-Z]{2,8}\d{2,8})', r'(\d{6})']
        for pattern in style_patterns:
            match = re.search(pattern, title_text)
            if match:
                info['style_number'] = match.group(1)
                break
        
        if info['style_number']:
            remainder = title_text.split(info['style_number'], 1)[-1]
            product_name_candidate = re.split(r'[ -]+[A-Z\s]+-\d{3}', remainder, 1)[0]
            info['product_name'] = product_name_candidate.strip(' -–—â€"').strip()

        return info

    def extract_brand(chunks):
        """
        FINAL CORRECTED VERSION: Specifically parses the <::logo: ... ::> block.
        """
        logo_chunk = next((c for c in chunks if c.get('type') == 'logo'), None)
        if not logo_chunk: return None

        raw_markdown = logo_chunk.get('markdown', '')
        
        # Use regex to find content INSIDE the <::logo: ... ::> block
        match = re.search(r'<::logo:(.*)::>', raw_markdown, re.DOTALL)
        if not match:
            return None # No logo block found

        # Extract the inner text and clean it
        inner_text = match.group(1).strip()
        
        # The first line of the inner text is almost always the brand
        for line in inner_text.splitlines():
            brand_candidate = line.strip()
            if brand_candidate:
                return brand_candidate # Return the first non-empty line
        
        return None

    def extract_colorways(chunks, style_number):
        """Ported from extractColorways in JS script, with de-duplication."""
        colorway_map = {} # Use dict for de-duplication
        text_chunks = [c for c in chunks if c.get('type') == 'text']

        for chunk in text_chunks:
            text = clean_html(chunk.get('markdown', ''))
            pattern = re.compile(r'([A-Z\s/]+?)\s*-\s*(\d{3})\s*-\s*\(PANTONE\s*([^)]+)\)')
            for match in pattern.finditer(text):
                color_name, color_number, pantone = match.groups()
                color_name = color_name.strip()
                
                if color_number not in colorway_map:
                    colorway_map[color_number] = {
                        'colorName': color_name,
                        'colorNumber': color_number,
                        'pantone': pantone.strip(),
                        'hex': '000000'
                    }
        return list(colorway_map.values()) if colorway_map else []

    def extract_measurements(chunks):
        """Refined to filter out junk header rows."""
        measurements = []
        table_chunks = [c for c in chunks if c.get('type') == 'table']
        for chunk in table_chunks:
            html = chunk.get('markdown', '')
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL)
            for row in rows:
                cells = [clean_html(c) for c in re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)]
                
                if not cells or not cells[0] or len(cells[0]) > 5 or "logo" in cells[0].lower():
                    continue
                
                pom = {
                    "code": cells[0],
                    "name": cells[1] if len(cells) > 1 else '',
                    "tolerance_negative": cells[3] if len(cells) > 3 else '0',
                    "tolerance_positive": cells[4] if len(cells) > 4 else '0',
                    "base_size_value": cells[-1] if cells else '0',
                    "comment": cells[2] if len(cells) > 2 else '',
                    "graded_values": ""
                }
                measurements.append(pom)
        return measurements
        
    # --- Main Transformation Flow ---
    title_info = extract_title_info(chunks)
    brand = extract_brand(chunks)
    colorways = extract_colorways(chunks, title_info.get('style_number', ''))
    measurements = extract_measurements(chunks)

    # Build the final structure to match the target JSON
    output = {
        "product_information": {
            "style_number": title_info.get('style_number') or "UNKNOWN",
            "product_name": title_info.get('product_name') or "UNKNOWN",
            "brand": brand or "UNKNOWN",
            "core_size_range": "---", "core_main_material": "", "product_type": "Short",
            "delivery": "", "gender": "Men", "product_category": "", "season_year": "Spring '26",
            "year": "2026", "season": "Spring", "fabric_group": "Knit", "classification": "Active",
            "brand_logo": ""
        },
        "colorways": [
            {
                "id": "", "imageHeaderId": None, "unlinkImage": False,
                "front_image": f"extracted_images/colorway_placeholder_{cw['colorNumber']}.png",
                "fields": [
                    {"id": "colorNumber", "value": cw['colorNumber']},
                    {"id": "colorName", "value": cw['colorName']},
                    {"id": "primary", "value": cw['hex']},
                    {"id": "brand_marketing_name", "value": f"{cw['colorNumber']} - {cw['colorName']}"},
                    {"id": "pantone", "value": cw['pantone']},
                    {"id": "secondary", "value": ""},
                    {"id": "color_number", "value": cw['colorNumber']},
                    {"id": "color_name", "value": cw['colorName']}
                ]
            } for cw in colorways
        ],
        "measurements": {"poms": measurements},
        "construction": {"details": "", "images": []},
        "sizeClasses": [
            {
              "sizeClassName": "Standard",
              "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
              "sampleSize": "M"
            }
        ],
        "front_image": ""
    }
    print("✅ Final Transformation complete.")
    return output

# --- IMAGE EXTRACTION & MAPPING (WITH FILTERING) ---
def extract_and_map_images(extraction_data, pdf_path, temp_dir, target_json):
    """
    Extracts images, maps them, AND filters out small, logo-like figures
    from the construction images list.
    """
    print("🖼️  Extracting and mapping images (with filtering)...")
    doc = fitz.open(pdf_path)
    relative_images_dir_name = "extracted_images"
    images_dir = Path(temp_dir) / relative_images_dir_name
    images_dir.mkdir(exist_ok=True)
    
    chunks = extraction_data.get('chunks', [])
    
    figure_chunks = [c for c in chunks if c.get('type') == 'figure']
    front_image_chunk = figure_chunks[0] if figure_chunks else None
    brand_logo_chunk = next((c for c in chunks if c.get('type') == 'logo'), None)
    
    construction_chunks = []
    processed_pages = set()
    for potential_fig_chunk in figure_chunks:
        page_num = potential_fig_chunk.get('grounding', {}).get('page')
        if page_num is None or page_num in processed_pages:
            continue

        page_text_chunks = [c for c in chunks if c.get('grounding', {}).get('page') == page_num and c.get('type') == 'text']
        page_text = " ".join([c.get('markdown', '') for c in page_text_chunks])
        
        if "CONSTRUCTION" in page_text.upper():
            figures_on_this_page = [fig for fig in figure_chunks if fig.get('grounding', {}).get('page') == page_num]
            
            for fig in figures_on_this_page:
                box = fig.get('grounding', {}).get('box')
                if not box: continue
                
                width = box['right'] - box['left']
                height = box['bottom'] - box['top']
                area = width * height
                
                if area < 0.03:
                    print(f"  🗑️  Discarding small figure (likely a logo) on construction page: {fig['id']}")
                    continue
                
                construction_chunks.append(fig)
            
            processed_pages.add(page_num)

    def save_chunk_image(chunk, prefix):
        if not chunk: return ""
        try:
            grounding = chunk.get('grounding', {})
            page_num = grounding.get('page')
            box = grounding.get('box')
            if page_num is None or not box: return ""

            page = doc[page_num]
            rect = fitz.Rect(box['left'] * page.rect.width, box['top'] * page.rect.height,
                             box['right'] * page.rect.width, box['bottom'] * page.rect.height)
            pix = page.get_pixmap(clip=rect, dpi=200)
            
            filename = f"{prefix}_{chunk['id']}.png"
            filepath = images_dir / filename
            pix.save(str(filepath))
            return f"{relative_images_dir_name}/{filename}"
        except Exception as e:
            print(f"  ⚠️  Could not save image for chunk {chunk.get('id')}: {e}")
            return ""

    target_json["front_image"] = save_chunk_image(front_image_chunk, "front")
    target_json["product_information"]["brand_logo"] = save_chunk_image(brand_logo_chunk, "logo")
    target_json["construction"]["images"] = [save_chunk_image(c, "construction") for c in construction_chunks]

    doc.close()
    print("✅ Image extraction and mapping complete.")
    return target_json

# --- MAIN ORCHESTRATOR ---
def process_tech_pack(pdf_path, temp_dir):
    """
    The main pipeline function that orchestrates the entire process.
    """
    api_key = os.getenv("VISION_AGENT_API_KEY")

    # 1. Parse with Landing AI
    extraction_data = parse_pdf_with_landingai(pdf_path, api_key)
    
    # 2. Transform text data (Final Corrected Version)
    final_json = transform_extraction_data(extraction_data)
    
    # 3. Extract and map images, updating the JSON in place
    final_json = extract_and_map_images(extraction_data, pdf_path, temp_dir, final_json)
    
    return final_json

