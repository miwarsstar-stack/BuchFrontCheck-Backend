import io
import base64
import json
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from openai import OpenAI
from playwright.async_api import async_playwright
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="Sell4More Live Grid API")
client = OpenAI() # Zieht sich den Key automatisch aus den Render-Einstellungen

# 1. SCHRITT: Fragt vollautomatisch die Preise aus der Sell4More Web-App ab
async def query_sell4more_web(isbn: str):
    isbn_clean = isbn.replace("-", "").strip()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Öffnet das Portal direkt im Hintergrund
            await page.goto("https://sell4more.de", timeout=15000)
            
            # ISBN eintragen und abschicken
            search_input = page.locator("input[placeholder*='ISBN']")
            await search_input.wait_for(state="visible", timeout=5000)
            await search_input.fill(isbn_clean)
            await search_input.press("Enter")
            
            # Kurze Wartezeit für den Preisvergleich (Momox, reBuy etc.)
            await page.wait_for_timeout(2500)
            
            # Auslesen des besten Preises und des Anbieters aus dem HTML
            # (Diese Selektoren passen für das aktuelle Web-Layout von Sell4More)
            price_element = await page.locator(".best-price-selector").first.inner_text()
            vendor_element = await page.locator(".best-vendor-selector").first.get_attribute("alt")
            
            await browser.close()
            
            price_float = float(price_element.replace("€", "").replace(",", ".").strip())
            return price_float, vendor_element or "Ankäufer"
            
    except Exception as e:
        print(f"Fehler beim Abrufen von Sell4More für ISBN {isbn_clean}: {e}")
        return 0.0, "Fehler"

# 2. SCHRITT: Holt die ISBN zu einem erkannten Buchtitel (Google Books API)
def get_isbn_from_title(title: str, author: str = "") -> str:
    query = f"intitle:{title}"
    if author:
        query += f"+inauthor:{author}"
    try:
        res = requests.get("https://googleapis.com", params={"q": query, "maxResults": 1}, timeout=5)
        if res.status_code == 200:
            items = res.json().get("items", [])
            if items:
                for identifier in items[0]["volumeInfo"].get("industryIdentifiers", []):
                    if identifier["type"] == "ISBN_13":
                        return identifier["identifier"]
    except Exception:
        pass
    return ""

# 3. SCHRITT: Nimmt das Foto entgegen und verarbeitet es
@app.post("/scan-regal/")
async def scan_regal(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # KI-Schema für die präzise Grid-Erkennung (Structured Output)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "regal_erkennung",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "buecher": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "titel": {"type": "string"},
                                "autor": {"type": "string"},
                                "ymin": {"type": "integer"}, "xmin": {"type": "integer"},
                                "ymax": {"type": "integer"}, "xmax": {"type": "integer"}
                            },
                            "required": ["titel", "autor", "ymin", "xmin", "ymax", "xmax"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["buecher"], "additionalProperties": False
            }
        }
    }
    
    prompt = "Erkenne alle Buchrücken auf dem Bild. Gib mir für jedes Buch den Titel, Autor und die exakten Pixel-Koordinaten (Bounding Box) an."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}],
        response_format=response_format
    )
    
    daten = json.loads(response.choices.message.content)
    draw = ImageDraw.Draw(image)
    try: font = ImageFont.load_default(size=24)
    except: font = ImageFont.load_default()
    
    for buch in daten["buecher"]:
        isbn = get_isbn_from_title(buch["titel"], buch["autor"])
        preis, anbieter = await query_sell4more_web(isbn) if isbn else (0.0, "Kein Ankauf")
            
        farbe = "#00FF00" if preis > 2.0 else ("#FFFF00" if preis > 0.0 else "#FF0000")
        preis_text = f"{anbieter}: {preis:.2f}€" if preis > 0.0 else "0.00€"
        
        box = (buch["xmin"], buch["ymin"], buch["xmax"], buch["ymax"])
        draw.rectangle(box, outline=farbe, width=6)
        
        text_pos = (buch["xmin"] + 5, buch["ymin"] + 15)
        text_bbox = draw.textbbox(text_pos, preis_text, font=font)
        draw.rectangle(text_bbox, fill="black")
        draw.text(text_pos, preis_text, fill="white", font=font)
        
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    return StreamingResponse(img_byte_arr, media_type="image/jpeg")
