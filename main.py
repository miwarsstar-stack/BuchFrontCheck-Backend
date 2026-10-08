import io
import base64
import json
import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from playwright.async_api import async_playwright
from PIL import Image, ImageDraw, ImageFont

app = FastAPI(title="Sell4More Live Grid API")
client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Erlaubt FlutLab den Zugriff
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. SCHRITT: Playwright-Abfrage für die Sell4More Web-App
async def query_sell4more_web(isbn: str):
    isbn_clean = isbn.replace("-", "").strip()
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://sell4more.de", timeout=15000)
            
            search_input = page.locator("input[placeholder*='ISBN']")
            await search_input.wait_for(state="visible", timeout=5000)
            await search_input.fill(isbn_clean)
            await search_input.press("Enter")
            
            await page.wait_for_timeout(2500)
            
            price_element = await page.locator(".best-price-selector").first.inner_text()
            vendor_element = await page.locator(".best-vendor-selector").first.get_attribute("alt")
            
            await browser.close()
            
            price_float = float(price_element.replace("€", "").replace(",", ".").strip())
            return price_float, vendor_element or "Ankäufer"
    except Exception as e:
        print(f"Fehler bei Sell4More Web-App: {e}")
        return 0.0, "Fehler"

# 2. SCHRITT: Titel in ISBN umwandeln
def get_isbn_from_title(title: str, author: str = "") -> str:
    query = f"intitle:{title}"
    if author:
        query += f"+inauthor:{author}"
    try:
        res = requests.get("https://www.googleapis.com/books/v1/volumes", params={"q": query, "maxResults": 1}, timeout=5)
        if res.status_code == 200:
            items = res.json().get("items", [])
            if items:
               for identifier in items[0]["volumeInfo"].get("industryIdentifiers", []):
                    if identifier["type"] == "ISBN_13":
                        return identifier["identifier"]
    except Exception:
        pass
    return ""

# 3. SCHRITT: Das Bild-Verarbeitungs-Gehirn
def verarbeite_das_bild(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
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
                                "titel": {"type": "string"}, "autor": {"type": "string"},
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
    
    prompt = "Erkenne alle Buchrücken auf dem Bild. Gib mir für jedes Buch den Titel, Autor und die exakten Pixel-Koordinaten an."
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}],
        response_format=response_format
    )
    
    daten = json.loads(response.choices[0].message.content)
    draw = ImageDraw.Draw(image)
    try: font = ImageFont.load_default(size=24)
    except: font = ImageFont.load_default()
    
    # Schleife wird asynchron aufgerufen, da query_sell4more_web eine async-Funktion ist
    return image, daten, draw, font

# 4. SCHRITT: DIE RADIKALE ANPASSUNG - BEIDE ENDPOINTS NEHMEN BILDER AN!

@app.post("/")
async def scan_regal_root(file: UploadFile = File(...)):
    """Nimmt das Bild an, falls das Handy stur auf die Startseite postet"""
    image_bytes = await file.read()
    image, daten, draw, font = verarbeite_das_bild(image_bytes)
    
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

@app.post("/scan-regal")
async def scan_regal_endpoint(file: UploadFile = File(...)):
    """Klassischer Endpoint als Backup"""
    return await scan_regal_root(file)

@app.get("/")
async def health_check():
    """Zeigt an, ob der Server wach ist"""
    return {"status": "online", "info": "Sende dein Bild per POST direkt hierhin!"}

@app.head("/")
async def head_fallback():
    return {}
